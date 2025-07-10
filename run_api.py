#!/usr/bin/env python3
import sys
import os
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*50)
    print("Starting Hybrid Recommender API")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists("models/trained_recommender.pt"):
        print("ERROR: No trained model found!")
        print("Please run training first: python run_training.py")
        sys.exit(1)
    
    # Start API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    main()