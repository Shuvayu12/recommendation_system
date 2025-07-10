#!/usr/bin/env python3
import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_recommender
from config import config

def main():
    print("="*50)
    print("Starting Hybrid Recommender Training")
    print("="*50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Start training
        recommender = train_recommender()
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Model saved to: models/trained_recommender.pt")
        print("="*50)
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()