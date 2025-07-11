import os
import requests
import zipfile
from pathlib import Path

def setup_project():
    """Set up the project structure"""
    # Create directories
    directories = [
        'models',
        'dataset/train',
        'test_images',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Project structure created successfully!")
    print("\nNext steps:")
    print("1. Download datasets from Kaggle (see links below)")
    print("2. Organize your dataset in the dataset/train/ folder")
    print("3. Run train_classifier.py to train your model")
    print("4. Test with smart_assistant.py")

if __name__ == "__main__":
    setup_project()
