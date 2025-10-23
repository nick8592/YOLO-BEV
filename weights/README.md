# Weights Directory

This directory contains pre-trained model weights and checkpoints.

## Contents

- `yolov8n.pt` - Pre-trained YOLOv8 nano model weights (6.2MB)
- Future: Custom trained 3D estimator weights will be stored here

## Usage

The YOLO detector automatically loads weights from this directory:
```python
from src.models.yolo_detector import YOLODetector

detector = YOLODetector(model_path='weights/yolov8n.pt')
```

## Download Additional Weights

```bash
# YOLOv8 models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -P weights/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P weights/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt -P weights/
```

## .gitignore

Large weight files (>10MB) are excluded from git by default.
