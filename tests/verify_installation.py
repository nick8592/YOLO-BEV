#!/usr/bin/env python3
"""
Quick verification test for YOLO-BEV installation
"""

import sys
from pathlib import Path

print("=" * 60)
print("YOLO-BEV Installation Verification")
print("=" * 60)

# Test 1: Basic imports
print("\n1. Testing basic imports...")
try:
    import torch
    import torchvision
    import numpy as np
    import cv2
    import yaml
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ NumPy {np.__version__}")
    print(f"   ✓ OpenCV {cv2.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: YOLO import
print("\n2. Testing YOLO (Ultralytics)...")
try:
    from ultralytics import YOLO
    print(f"   ✓ Ultralytics YOLO imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: nuScenes devkit
print("\n3. Testing nuScenes devkit...")
try:
    from nuscenes.nuscenes import NuScenes
    print(f"   ✓ nuScenes devkit imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Project modules
print("\n4. Testing project modules...")
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.models.yolo_detector import YOLODetector
    print(f"   ✓ YOLODetector imported")
except Exception as e:
    print(f"   ✗ YOLODetector error: {e}")

try:
    from src.models.estimator_3d import Estimator3D
    print(f"   ✓ Estimator3D imported")
except Exception as e:
    print(f"   ✗ Estimator3D error: {e}")

try:
    from src.utils.bev_transform import BEVTransform
    print(f"   ✓ BEVTransform imported")
except Exception as e:
    print(f"   ✗ BEVTransform error: {e}")

try:
    from src.utils.visualization import Visualizer
    print(f"   ✓ Visualizer imported")
except Exception as e:
    print(f"   ✗ Visualizer error: {e}")

# Test 5: Configuration
print("\n5. Testing configuration...")
try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ Configuration loaded successfully")
    print(f"   - Dataset: {config['dataset']['name']}")
    print(f"   - YOLO model: {config['yolo']['model_name']}")
    print(f"   - 3D backbone: {config['estimation_3d']['backbone']}")
except Exception as e:
    print(f"   ✗ Configuration error: {e}")

# Test 6: Check nuScenes data
print("\n6. Checking nuScenes data...")
try:
    import os
    nuscenes_path = 'nuscenes/samples/CAM_FRONT'
    if os.path.exists(nuscenes_path):
        images = list(Path(nuscenes_path).glob('*.jpg'))
        print(f"   ✓ nuScenes data found: {len(images)} images in CAM_FRONT")
    else:
        print(f"   ⚠ nuScenes data not found at {nuscenes_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("Installation Verification Complete!")
print("=" * 60)
print("\nYou can now run:")
print("  - python3 scripts/demo.py          (for quick demo)")
print("  - python3 scripts/inference.py     (for custom inference)")
print("=" * 60)
