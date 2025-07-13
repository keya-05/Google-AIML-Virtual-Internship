import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector with pre-trained model"""
        print("Loading object detection model...")
        # Using SSD MobileNet v2 which is more compatible and stable
        self.detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        print("Model loaded successfully!")
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, tf.Tensor]:
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor with uint8 dtype (required by TensorFlow Hub models)
        input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        return image, input_tensor
    
    def detect_objects(self, image_path: str, confidence_threshold: float = 0.3) -> List[dict]:
        """Detect objects in image and return bounding boxes"""
        try:
            image, input_tensor = self.load_image(image_path)
            
            # Run detection
            detections = self.detector(input_tensor)
            
            # Extract results (SSD MobileNet v2 format)
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(int)
            
            # Filter by confidence and get valid detections
            valid_detections = []
            h, w = image.shape[:2]
            
            for i in range(len(scores)):
                if scores[i] > confidence_threshold:
                    # Boxes are in format [y1, x1, y2, x2] normalized
                    y1, x1, y2, x2 = boxes[i]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    # Skip if bounding box is too small
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    
                    # Crop the detected object
                    cropped_object = image[y1:y2, x1:x2]
                    
                    # Skip if cropped image is empty
                    if cropped_object.size == 0:
                        continue
                    
                    valid_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(scores[i]),
                        'class_id': int(classes[i]),
                        'cropped_image': cropped_object
                    })
            
            return valid_detections
            
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return []
    
    def visualize_detections(self, image_path: str, detections: List[dict], save_path: str = None):
        """Visualize detection results"""
        try:
            image, _ = self.load_image(image_path)
            
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(x1, y1 - 10, f'Object {i+1}: {confidence:.2f}',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax.set_title('Object Detection Results')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")

# Test function
def test_detector():
    """Test the object detector"""
    detector = ObjectDetector()
    
    # Test with a sample image
    test_image = "test_images/desk_sample.jpg"
    
    if tf.io.gfile.exists(test_image):
        detections = detector.detect_objects(test_image)
        print(f"Found {len(detections)} objects")
        
        for i, detection in enumerate(detections):
            print(f"Object {i+1}: Confidence {detection['confidence']:.2f}")
        
        # Visualize results
        detector.visualize_detections(test_image, detections)
    else:
        print(f"Test image not found: {test_image}")

if __name__ == "__main__":
    test_detector()
