# Trash Detection for Raspberry Pi 5

YOLOv8n TensorFlow Lite implementation for real-time trash detection on Raspberry Pi 5 with OV5647 camera module.

## Project Structure

```
trash_detection/
├── model/
│   ├── model_float32.tflite  # YOLOv8n TensorFlow Lite model
│   └── labels.txt            # Class labels (one per line)
├── ai.py                     # Main inference module (headless, importable)
├── test_camera.py            # Visual test script with real-time preview
└── README.md                 # This file
```

## Classes

The model detects exactly 3 classes (in this order):
1. crumpled paper
2. disposable cup
3. plastic bottle

## Requirements

### System Requirements
- Raspberry Pi 5 running Debian Trixie (64-bit)
- OV5647 camera module connected via rpicam interface
- Python 3.8 or higher

### Python Dependencies

Install the required packages:

```bash
# For TensorFlow Lite (preferred - lighter weight)
pip install tflite-runtime

# OR use full TensorFlow (if tflite-runtime is not available)
pip install tensorflow

# Camera and image processing
pip install picamera2 opencv-python numpy
```

### Full Installation Command

```bash
pip install tflite-runtime picamera2 opencv-python numpy
```

**Note:** On Raspberry Pi, you may need to install `tflite-runtime` from a wheel file or build it. Alternatively, use the full `tensorflow` package.

## Setup

1. **Place your model files:**
   - Copy `model_float32.tflite` to `trash_detection/model/`
   - Ensure `labels.txt` exists in `trash_detection/model/` with the 3 class names (one per line)

2. **Verify camera access:**
   ```bash
   # Test camera with libcamera
   libcamera-hello --list-cameras
   ```

## Usage

### Main Detection Module (`ai.py`)

The `ai.py` module is designed to be headless and importable for integration into larger systems.

**Standalone execution:**
```bash
cd trash_detection
python3 ai.py
```

**As a module:**
```python
from ai import TrashDetector

# Initialize detector
detector = TrashDetector()

# Perform detection (captures frame automatically)
detections = detector.detect()

# Process results
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
    print(f"Bbox: {det['bbox']}")

# Cleanup
detector.cleanup()
```

**Detection output format:**
Each detection is a dictionary with:
- `class_id`: Integer class ID (0-2)
- `class_name`: String class name
- `confidence`: Float confidence score (0.0-1.0)
- `bbox`: List `[x, y, w, h]` in pixel coordinates

### Visual Test Script (`test_camera.py`)

Run the visual test script for real-time preview with bounding boxes:

```bash
cd trash_detection
python3 test_camera.py
```

- Press `q` to quit
- Displays real-time camera feed with detections overlaid
- Shows bounding boxes, class names, and confidence scores

## Code Features

### `ai.py` - Main Module
- **Headless operation**: No GUI dependencies
- **Importable**: Designed for integration into SLAM/navigation systems
- **Efficient preprocessing**: Resizes to 640x640, normalizes to [0,1]
- **YOLOv8 post-processing**: Handles [1, 84, 8400] output tensor
- **Non-Maximum Suppression (NMS)**: Removes overlapping detections
- **Configurable confidence threshold**: Default 0.5
- **Proper resource cleanup**: Camera cleanup methods

### `test_camera.py` - Test Script
- Real-time camera preview
- Visual bounding box overlay
- Performance testing
- Self-contained (can be deleted later)

## Technical Details

### Model Input/Output
- **Input**: RGB image, 640x640 pixels, normalized to [0.0, 1.0]
- **Output**: Tensor shape [1, 84, 8400]
  - 84 = 4 (bbox coords) + 80 (or num_classes for custom models)
  - 8400 = number of anchor points

### Post-Processing Pipeline
1. Extract bounding box coordinates (normalized center-based format)
2. Extract class scores and apply softmax
3. Filter by confidence threshold (default: 0.5)
4. Convert to pixel coordinates (x, y, w, h)
5. Apply Non-Maximum Suppression (IoU threshold: 0.45)

### Camera Configuration
- Uses `picamera2` (not legacy `picamera`)
- Compatible with rpicam/libcamera stack
- Default resolution: 1640x1232 (OV5647 native)
- RGB888 format

## Integration Notes

The `ai.py` module is designed for integration into larger robotic systems:

- **SLAM/Navigation**: Import `TrashDetector` class and use in your navigation loop
- **No side effects on import**: All execution logic wrapped in `if __name__ == "__main__":`
- **Efficient**: Optimized for Raspberry Pi 5 performance
- **Modular**: Clear separation of concerns (preprocessing, inference, post-processing)

Example integration:
```python
from ai import TrashDetector

class NavigationSystem:
    def __init__(self):
        self.detector = TrashDetector()
    
    def detect_trash(self):
        detections = self.detector.detect()
        # Process detections for navigation/LIDAR fusion
        return detections
```

## Troubleshooting

### Camera not detected
- Ensure camera is connected and enabled in `raspi-config`
- Check with `libcamera-hello --list-cameras`

### Model not found
- Verify `model_float32.tflite` is in `trash_detection/model/`
- Check file permissions

### Import errors
- Install missing dependencies: `pip install <package-name>`
- For `tflite_runtime`, you may need to install from a wheel file on Raspberry Pi

### Low performance
- Reduce detection frequency in test script
- Lower camera resolution if needed
- Adjust confidence threshold

## License

This code is provided as-is for use with YOLOv8n models on Raspberry Pi 5.

