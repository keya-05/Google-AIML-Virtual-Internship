import os
import cv2
import numpy as np
from object_detector import ObjectDetector
from desk_classifier import DeskClassifier
from typing import List, Dict
import json
from datetime import datetime

class SmartDeskAssistant:
    def __init__(self):
        """Initialize the Smart Desk Assistant"""
        print("Initializing Smart Desk Assistant...")
        self.detector = ObjectDetector()
        self.classifier = DeskClassifier()
        print("Smart Desk Assistant ready!")
    
    def analyze_desk(self, image_path: str, confidence_threshold: float = 0.3) -> Dict:
        """Analyze desk image and return detected and classified items"""
        print(f"Analyzing desk image: {image_path}")
        
        # Detect objects
        detections = self.detector.detect_objects(image_path, confidence_threshold)
        print(f"Found {len(detections)} objects")
        
        # Classify detected objects
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'total_objects': len(detections),
            'items': []
        }
        
        for i, detection in enumerate(detections):
            cropped_image = detection['cropped_image']
            class_name, class_confidence = self.classifier.classify_image(cropped_image)
           
            item_info = {
                'item_id': i + 1,
                'detected_class': class_name,
                'classification_confidence': float(class_confidence),
                'detection_confidence': float(detection['confidence']),
                'bbox': detection['bbox']
            }
            
            results['items'].append(item_info)
            print(f"Item {i+1}: {class_name} (confidence: {class_confidence:.2f})")
        
        return results
    
    def save_results(self, results: Dict, output_path: str = 'results.json'):
        """Save analysis results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def create_annotated_image(self, image_path: str, results: Dict, output_path: str = 'annotated_desk.jpg'):
        """Create annotated image with detected items"""
        image = cv2.imread(image_path)
        
        for item in results['items']:
            x1, y1, x2, y2 = item['bbox']
            class_name = item['detected_class']
            confidence = item['classification_confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved to {output_path}")

def main():
    """Main function to run the Smart Desk Assistant"""
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    
    # Initialize assistant
    assistant = SmartDeskAssistant()
    
    # Analyze desk image
    image_path = 'test_images/desk_sample.jpg'  # Replace with your image path
    
    if os.path.exists(image_path):
        results = assistant.analyze_desk(image_path)
        
        # Save results
        assistant.save_results(results, 'results/analysis_results.json')
        
        # Create annotated image
        assistant.create_annotated_image(image_path, results, 'results/annotated_desk.jpg')
        
        # Print summary
        print("\n=== DESK ANALYSIS SUMMARY ===")
        print(f"Total objects detected: {results['total_objects']}")
        
        item_counts = {}
        for item in results['items']:
            class_name = item['detected_class']
            item_counts[class_name] = item_counts.get(class_name, 0) + 1
        
        print("\nItem inventory:")
        for item_type, count in item_counts.items():
            print(f"  {item_type}: {count}")
    
    else:
        print(f"Image not found: {image_path}")
        print("Please place a desk image in the test_images folder")

if __name__ == "__main__":
    main()

