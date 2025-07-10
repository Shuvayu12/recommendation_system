from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import torch
import uvicorn
import logging
import asyncio
from contextlib import asynccontextmanager
import os

from models.hybrid import HybridRecommender
from models.unlearning import UnlearningManager
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "recommender": None,
    "unlearning_manager": None,
    "is_ready": False
}

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID for recommendations")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations")
    filter_seen: bool = Field(True, description="Filter out already seen items")

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, float]]
    total_items: int
    filtered_items: int = 0

class UnlearningRequest(BaseModel):
    request_type: str = Field(..., regex="^(user|item|interaction)$")
    user_id: Optional[int] = Field(None, description="User ID for user unlearning")
    item_id: Optional[int] = Field(None, description="Item ID for item unlearning")
    interaction_id: Optional[int] = Field(None, description="Interaction ID for interaction unlearning")
    
    class Config:
        schema_extra = {
            "example": {
                "request_type": "user",
                "user_id": 123
            }
        }

class UnlearningResponse(BaseModel):
    message: str
    request_type: str
    processed_requests: int
    pending_requests: int

class SystemStatus(BaseModel):
    status: str
    uptime: str
    model_info: Dict
    stats: Dict

def load_recommender_safely():
    """Safely load recommender with error handling"""
    try:
        model_path = "models/trained_recommender.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=config.device)
        
        # Recreate recommender
        recommender = HybridRecommender(checkpoint['config'])
        
        # Restore state
        recommender.user_map = checkpoint['user_map']
        recommender.item_map = checkpoint['item_map']
        recommender.item_content_features = checkpoint['item_content_features'].to(config.device)
        recommender.graph = {k: v.to(config.device) for k, v in checkpoint['graph'].items()}
        
        # Load model state
        recommender.model.load_state_dict(checkpoint['model_state_dict'])
        recommender.model.eval()
        
        # Initialize reverse mappings
        recommender.reverse_user_map = {v: k for k, v in recommender.user_map.items()}
        recommender.reverse_item_map = {v: k for k, v in recommender.item_map.items()}
        
        logger.info(f"Recommender loaded successfully from {model_path}")
        logger.info(f"Model timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        logger.info(f"Training loss: {checkpoint.get('loss', 'Unknown')}")
        
        return recommender
        
    except Exception as e:
        logger.error(f"Failed to load recommender: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting recommendation service...")
    try:
        app_state["recommender"] = load_recommender_safely()
        app_state["unlearning_manager"] = UnlearningManager(config)
        app_state["is_ready"] = True
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        app_state["is_ready"] = False
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down recommendation service...")
    app_state["is_ready"] = False

# Create FastAPI app
app = FastAPI(
    title="Hybrid Recommendation System",
    description="A recommendation system with machine unlearning capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_service_ready():
    """Check if service is ready"""
    if not app_state["is_ready"] or app_state["recommender"] is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready. Please wait for model loading to complete."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if app_state["is_ready"] else "loading",
        "timestamp": torch.utils.data.get_worker_info()
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user"""
    validate_service_ready()
    
    recommender = app_state["recommender"]
    
    # Validate user exists
    if request.user_id not in recommender.user_map:
        raise HTTPException(
            status_code=404, 
            detail=f"User {request.user_id} not found"
        )
    
    try:
        # Get recommendations
        recommendations = recommender.recommend(
            request.user_id, 
            request.k,
            filter_seen=request.filter_seen
        )
        
        # Format response
        formatted_recs = [
            {"item_id": int(item_id), "score": float(score)}
            for item_id, score in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=formatted_recs,
            total_items=len(recommendations),
            filtered_items=0  # TODO: Implement filtering count
        )
        
    except Exception as e:
        logger.error(f"Recommendation error for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unlearn", response_model=UnlearningResponse)
async def request_unlearning(request: UnlearningRequest, background_tasks: BackgroundTasks):
    """Submit an unlearning request"""
    validate_service_ready()
    
    # Validate request
    if request.request_type == 'user' and request.user_id is None:
        raise HTTPException(status_code=400, detail="user_id required for user unlearning")
    elif request.request_type == 'item' and request.item_id is None:
        raise HTTPException(status_code=400, detail="item_id required for item unlearning")
    elif request.request_type == 'interaction' and request.interaction_id is None:
        raise HTTPException(status_code=400, detail="interaction_id required for interaction unlearning")
    
    unlearning_manager = app_state["unlearning_manager"]
    
    try:
        # Register request
        unlearning_manager.register_request(
            request.request_type,
            request.user_id,
            request.item_id,
            request.interaction_id
        )
        
        # Process in background for better performance
        background_tasks.add_task(
            process_unlearning_background,
            unlearning_manager,
            app_state["recommender"]
        )
        
        pending_count = sum(
            1 for req in unlearning_manager.unlearning_requests 
            if not req.get('processed', False)
        )
        
        return UnlearningResponse(
            message=f"Unlearning request registered and will be processed in background",
            request_type=request.request_type,
            processed_requests=0,
            pending_requests=pending_count
        )
        
    except Exception as e:
        logger.error(f"Unlearning request error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_unlearning_background(unlearning_manager, recommender):
    """Background task for processing unlearning requests"""
    try:
        processed = unlearning_manager.process_pending_requests(recommender)
        logger.info(f"Processed {processed} unlearning requests")
    except Exception as e:
        logger.error(f"Background unlearning processing failed: {str(e)}")

@app.get("/status", response_model=SystemStatus)
async def system_status():
    """Get comprehensive system status"""
    validate_service_ready()
    
    recommender = app_state["recommender"]
    unlearning_manager = app_state["unlearning_manager"]
    
    # Calculate model info
    model_params = sum(p.numel() for p in recommender.model.parameters())
    model_size_mb = model_params * 4 / (1024 * 1024)  # Assuming float32
    
    return SystemStatus(
        status="operational",
        uptime="Unknown",  # TODO: Track actual uptime
        model_info={
            "device": str(next(recommender.model.parameters()).device),
            "parameters": model_params,
            "size_mb": round(model_size_mb, 2),
            "precision": "float32"
        },
        stats={
            "num_users": len(recommender.user_map),
            "num_items": len(recommender.item_map),
            "num_interactions": getattr(recommender, 'interaction_count', 0),
            "num_snapshots": len(getattr(recommender, 'snapshot_manager', {}).get('snapshots', {})),
            "pending_unlearning_requests": sum(
                1 for req in unlearning_manager.unlearning_requests 
                if not req.get('processed', False)
            )
        }
    )

@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Get user profile information"""
    validate_service_ready()
    
    recommender = app_state["recommender"]
    
    if user_id not in recommender.user_map:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # TODO: Implement user profile retrieval
    return {
        "user_id": user_id,
        "internal_id": recommender.user_map[user_id],
        "message": "User profile endpoint not yet implemented"
    }

@app.get("/items/{item_id}/info")
async def get_item_info(item_id: int):
    """Get item information"""
    validate_service_ready()
    
    recommender = app_state["recommender"]
    
    if item_id not in recommender.item_map:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    internal_id = recommender.item_map[item_id]
    
    return {
        "item_id": item_id,
        "internal_id": internal_id,
        "features_shape": list(recommender.item_content_features[internal_id].shape),
        "message": "Item details endpoint not yet implemented"
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )