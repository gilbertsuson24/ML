"""
YOLOv8n TensorFlow Lite inference module for Raspberry Pi 5.
Headless detection module - no GUI, designed for integration into larger systems.
"""

import os
import numpy as np
from pathlib import Path
import logging

# Try to import tensorflow.lite first, fallback to tflite_runtime
try:
    from tensorflow.lite.python.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        raise ImportError(
            "Neither tensorflow nor tflite_runtime is installed. "
            "Install with: pip install tensorflow or pip install tflite-runtime"
        )

try:
    from picamera2 import Picamera2
except ImportError:
    raise ImportError(
        "picamera2 is not installed. Install with: pip install picamera2"
    )

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python is not installed. Install with: pip install opencv-python"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrashDetector:
    """
    YOLOv8n-based trash detection using TensorFlow Lite.
    
    Handles model loading, camera initialization, inference, and post-processing.
    Designed for headless operation on Raspberry Pi 5 with OV5647 camera.
    """
    
    def __init__(self, model_path=None, labels_path=None, conf_threshold=0.5):
        """
        Initialize the trash detector.
        
        Args:
            model_path: Path to model_float32.tflite (default: model/model_float32.tflite)
            labels_path: Path to labels.txt (default: model/labels.txt)
            conf_threshold: Confidence threshold for detections (default: 0.5)
        """
        # Get the directory of this file
        script_dir = Path(__file__).parent
        
        # Set default paths relative to script directory
        if model_path is None:
            model_path = script_dir / "model" / "model_float32.tflite"
        else:
            model_path = Path(model_path)
        
        if labels_path is None:
            labels_path = script_dir / "model" / "labels.txt"
        else:
            labels_path = Path(labels_path)
        
        self.model_path = model_path
        self.labels_path = labels_path
        self.conf_threshold = conf_threshold
        
        # YOLOv8n input size
        self.input_size = 640
        
        # Load labels
        self.labels = self._load_labels()
        logger.info(f"Loaded {len(self.labels)} class labels")
        
        # Load TFLite model
        self.interpreter = self._load_model()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Input shape: [1, 640, 640, 3]
        input_shape = self.input_details[0]['shape']
        logger.info(f"Model input shape: {input_shape}")
        
        # Output shape: [1, 84, 8400] for YOLOv8n
        output_shape = self.output_details[0]['shape']
        logger.info(f"Model output shape: {output_shape}")
        
        # Initialize camera (will be done on first use)
        self.camera = None
        
    def _load_labels(self):
        """Load class labels from labels.txt file."""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        with open(self.labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        
        return labels
    
    def _load_model(self):
        """Load and initialize TensorFlow Lite model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        interpreter = Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        logger.info("Model loaded successfully")
        
        return interpreter
    
    def _initialize_camera(self):
        """Initialize the OV5647 camera using picamera2."""
        if self.camera is not None:
            return
        
        logger.info("Initializing camera...")
        self.camera = Picamera2()
        
        # Configure camera for RGB capture
        # Use a reasonable resolution (can be adjusted based on needs)
        camera_config = self.camera.create_preview_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}  # Common OV5647 resolution
        )
        self.camera.configure(camera_config)
        self.camera.start()
        
        # Give camera time to initialize
        import time
        time.sleep(2)
        
        logger.info("Camera initialized.")
    
    def _preprocess_image(self, image):
        """
        Preprocess image for YOLOv8 inference.
        
        Args:
            image: RGB numpy array (H, W, 3), values in [0, 255]
        
        Returns:
            Preprocessed image: (1, 640, 640, 3), float32, values in [0.0, 1.0]
        """
        # Resize to model input size (640x640)
        resized = cv2.resize(
            image, 
            (self.input_size, self.input_size), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert to float32 and normalize to [0.0, 1.0]
        # YOLOv8 float32 typically expects [0, 255] -> [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension: (1, 640, 640, 3)
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def _postprocess_output(self, output, original_shape):
        """
        Post-process YOLOv8 output tensor.
        
        YOLOv8 output shape: [1, 84, 8400]
        - 84 = 4 (bbox coords) + 80 (COCO classes) OR 4 + 3 (our 3 classes)
        - Actually, for custom models: 4 + num_classes
        - 8400 = number of anchors (80*80 + 40*40 + 20*20 = 8400 for YOLOv8n)
        
        Args:
            output: Raw model output [1, 84, 8400]
            original_shape: (height, width) of original image
        
        Returns:
            List of detections, each as dict with keys:
                'class_id', 'class_name', 'confidence', 'bbox' (x, y, w, h)
        """
        # Remove batch dimension: [84, 8400]
        output = output[0]
        
        # For YOLOv8, structure is:
        # output[0:4, :] = bbox coordinates (cx, cy, w, h) normalized to [0, 1]
        # output[4:, :] = class scores (logits)
        
        # Extract bbox coordinates (cx, cy, w, h) - normalized
        bboxes_cxcywh = output[0:4, :].T  # Shape: [8400, 4]
        
        # Extract class scores
        class_scores = output[4:, :].T  # Shape: [8400, num_classes]
        
        # If model has more classes than our labels, use only the first N classes
        # (where N = number of labels)
        num_model_classes = class_scores.shape[1]
        num_labels = len(self.labels)
        if num_model_classes > num_labels:
            # Use only the first N classes that match our labels
            class_scores = class_scores[:, :num_labels]
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(class_scores - np.max(class_scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Get max confidence and class for each anchor
        max_conf = np.max(probs, axis=1)  # [8400]
        class_ids = np.argmax(probs, axis=1)  # [8400]
        
        # Filter by confidence threshold
        mask = max_conf >= self.conf_threshold
        if not np.any(mask):
            return []
        
        # Get filtered results
        confidences = max_conf[mask]
        class_ids = class_ids[mask]
        bboxes_cxcywh = bboxes_cxcywh[mask]
        
        # Convert normalized (cx, cy, w, h) to pixel coordinates (x, y, w, h)
        orig_h, orig_w = original_shape[:2]
        
        # Scale factor for converting from 640x640 to original size
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        # Convert center-based to top-left corner based
        bboxes = []
        for cx_norm, cy_norm, w_norm, h_norm in bboxes_cxcywh:
            # Denormalize
            cx = cx_norm * self.input_size
            cy = cy_norm * self.input_size
            w = w_norm * self.input_size
            h = h_norm * self.input_size
            
            # Convert to (x, y, w, h) format
            x = (cx - w / 2) * scale_x
            y = (cy - h / 2) * scale_y
            w = w * scale_x
            h = h * scale_y
            
            bboxes.append([x, y, w, h])
        
        # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
        bboxes = np.array(bboxes)
        indices = self._nms(bboxes, confidences, iou_threshold=0.45)
        
        # Build results
        detections = []
        for idx in indices:
            class_id = int(class_ids[idx])
            # Safety check: ensure class_id is within label range
            if 0 <= class_id < len(self.labels):
                detections.append({
                    'class_id': class_id,
                    'class_name': self.labels[class_id],
                    'confidence': float(confidences[idx]),
                    'bbox': [float(x) for x in bboxes[idx]]
                })
        
        return detections
    
    def _nms(self, boxes, scores, iou_threshold=0.45):
        """
        Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            boxes: Array of bounding boxes [N, 4] in format (x, y, w, h)
            scores: Array of confidence scores [N]
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
        
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Calculate areas
        areas = boxes[:, 2] * boxes[:, 3]
        
        # Sort by score (descending)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # Take the box with highest score
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU < threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image=None):
        """
        Perform detection on a single frame.
        
        Args:
            image: Optional RGB numpy array. If None, captures from camera.
        
        Returns:
            List of detections (empty if none), each with keys:
                'class_id', 'class_name', 'confidence', 'bbox' (x, y, w, h)
        """
        # Capture frame if not provided
        if image is None:
            self._initialize_camera()
            image = self.camera.capture_array()
            logger.info("Captured frame.")
        
        # Store original shape
        original_shape = image.shape
        
        # Preprocess
        preprocessed = self._preprocess_image(image)
        
        # Set input tensor
        input_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_tensor_index, preprocessed)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_tensor_index = self.output_details[0]['index']
        output = self.interpreter.get_tensor(output_tensor_index)
        
        # Post-process
        detections = self._postprocess_output(output, original_shape)
        
        return detections
    
    def cleanup(self):
        """Clean up camera resources."""
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
            logger.info("Camera stopped.")


def main():
    """Main function for standalone execution."""
    try:
        detector = TrashDetector()
        
        # Capture and detect
        detections = detector.detect()
        
        if detections:
            for det in detections:
                bbox = det['bbox']
                print(f"Detection: {det['class_name']}, "
                      f"confidence: {det['confidence']:.2f}, "
                      f"bbox: [{int(bbox[0])}, {int(bbox[1])}, "
                      f"{int(bbox[2])}, {int(bbox[3])}]")
        else:
            print("No trash detected.")
        
        # Cleanup
        detector.cleanup()
        
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

