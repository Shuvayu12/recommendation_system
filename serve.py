from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
import uvicorn

from models.hybrid import HybridRecommender
from models.unlearning import UnlearningManager
from config import config

app = FastAPI()

# Global recommender instance
recommender: Optional[HybridRecommender] = None
unlearning_manager = UnlearningManager(config)

class RecommendationRequest(BaseModel):
    user_id: int
    k: int = 10

class UnlearningRequest(BaseModel):
    request_type: str  # 'user', 'item', or 'interaction'
    user_id: Optional[int] = None
    item_id: Optional[int] = None
    interaction_id: Optional[int] = None

@app.on_event("startup")
async def startup_event():
    """Load the recommender system on startup"""
    global recommender
    try:
        recommender = torch.load("trained_recommender.pt")
        recommender.config.device = config.device  # Update device in case it changed
        print("Recommender system loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load recommender system: {str(e)}")

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        recommendations = recommender.recommend(request.user_id, request.k)
        return {
            "user_id": request.user_id,
            "recommendations": [
                {"item_id": item_id, "score": float(score)}
                for item_id, score in recommendations
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unlearn")
async def request_unlearning(request: UnlearningRequest):
    """Submit an unlearning request"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    # Validate request
    if request.request_type == 'user' and request.user_id is None:
        raise HTTPException(status_code=400, detail="user_id required for user unlearning")
    if request.request_type == 'item' and request.item_id is None:
        raise HTTPException(status_code=400, detail="item_id required for item unlearning")
    if request.request_type == 'interaction' and request.interaction_id is None:
        raise HTTPException(status_code=400, detail="interaction_id required for interaction unlearning")
    
    # Register request
    unlearning_manager.register_request(
        request.request_type,
        request.user_id,
        request.item_id,
        request.interaction_id
    )
    
    # Process synchronously (for production, might want async processing)
    processed = unlearning_manager.process_pending_requests(recommender)
    
    return {
        "message": f"Unlearning request received. {processed} new requests processed.",
        "request_type": request.request_type,
        "user_id": request.user_id,
        "item_id": request.item_id,
        "interaction_id": request.interaction_id
    }

@app.get("/status")
async def system_status():
    """Get system status"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    return {
        "status": "operational",
        "num_users": len(recommender.user_map),
        "num_items": len(recommender.item_map),
        "num_interactions": recommender.interaction_count,
        "num_snapshots": len(recommender.snapshot_manager.snapshots),
        "pending_unlearning_requests": sum(
            1 for req in unlearning_manager.unlearning_requests if not req['processed'])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/model_info")
async def model_info():
    return {
        "device": str(recommender.model.device),
        "parameters": sum(p.numel() for p in recommender.model.parameters())
    }