"""
Visual test script for trash detection with real-time camera preview.
This script is for testing purposes only and can be deleted later.
"""

import cv2
import numpy as np
from ai import TrashDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_detections(image, detections):
    """
    Draw bounding boxes, labels, and confidence scores on image.
    
    Args:
        image: RGB numpy array (will be converted to BGR for cv2)
        detections: List of detection dictionaries
    
    Returns:
        Image with drawn detections
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for det in detections:
        bbox = det['bbox']
        x, y, w, h = [int(coord) for coord in bbox]
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Draw bounding box
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            image_bgr,
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            image_bgr,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return image_bgr


def main():
    """Main function for visual testing."""
    detector = None
    
    try:
        # Initialize detector
        logger.info("Initializing trash detector...")
        detector = TrashDetector()
        detector._initialize_camera()
        
        logger.info("Starting camera preview. Press 'q' to quit.")
        
        frame_count = 0
        
        while True:
            # Capture frame
            frame = detector.camera.capture_array()
            
            # Run detection every N frames (adjust for performance)
            # For real-time, you might want to detect every frame or every 2-3 frames
            if frame_count % 1 == 0:  # Detect every frame
                detections = detector.detect(image=frame)
            else:
                detections = []
            
            # Draw detections
            frame_with_detections = draw_detections(frame, detections)
            
            # Display frame
            cv2.imshow('Trash Detection - Press Q to quit', frame_with_detections)
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit requested by user.")
                break
            
            frame_count += 1
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if detector is not None:
            detector.cleanup()
        cv2.destroyAllWindows()
        logger.info("Test completed.")


if __name__ == "__main__":
    main()

