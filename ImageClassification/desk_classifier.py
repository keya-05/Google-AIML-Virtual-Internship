import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from typing import List, Tuple
import os

class DeskClassifier:
    def __init__(self, model_path: str = 'models/desk_classifier.h5'):
        """Initialize the classifier"""
        self.model = load_model(model_path)
        self.class_names = self._load_class_names()
        self.img_size = 224
    
    def _load_class_names(self) -> List[str]:
        """Load class names from file"""
        try:
            with open('models/class_names.txt', 'r') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            # Default classes if file not found
            return ['book', 'bottle', 'cup', 'laptop', 'phone']
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for classification"""
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert to array and normalize
        image = img_to_array(image)
        image = image.astype('float32') / 255.0
       # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def classify_image(self, image: np.ndarray) -> Tuple[str, float]:
        """Classify a single image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence
    
    def classify_multiple_images(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Classify multiple images"""
        results = []
        for image in images:
            class_name, confidence = self.classify_image(image)
            results.append((class_name, confidence))
        return results
