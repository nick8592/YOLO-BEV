#!/usr/bin/env python3
"""
Simple test of YOLO-BEV pipeline on a single image
"""

import os
import sys
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import YOLOBEVPipeline

print("="*60)
print("YOLO-BEV Pipeline Test")
print("="*60)

# Configuration
config_path = 'configs/config.yaml'
output_dir = 'outputs/test'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize pipeline
print("\n1. Initializing pipeline...")
try:
    pipeline = YOLOBEVPipeline(config_path=config_path)
    print("   ✓ Pipeline initialized successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Find a sample image
print("\n2. Loading sample image...")
nuscenes_path = 'nuscenes/samples/CAM_FRONT'
image_files = list(Path(nuscenes_path).glob('*.jpg'))

if not image_files:
    print(f"   ✗ No images found in {nuscenes_path}")
    sys.exit(1)

sample_image_path = str(image_files[0])
print(f"   Using: {Path(sample_image_path).name}")

# Load image
image = cv2.imread(sample_image_path)
if image is None:
    print(f"   ✗ Could not load image")
    sys.exit(1)

print(f"   ✓ Image loaded: {image.shape}")

# Process through pipeline
print("\n3. Processing through pipeline...")
try:
    print("   - Running 2D object detection (YOLO)...")
    results = pipeline.process_image(image)
    print(f"   ✓ Processing complete!")
except Exception as e:
    print(f"   ✗ Error during processing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Print results
print("\n4. Results:")
print(f"   - 2D Detections: {len(results['detections_2d']['boxes'])} objects")
print(f"   - 3D Boxes: {len(results['boxes_3d'])} objects")
print(f"   - BEV Boxes: {len(results['boxes_bev'])} objects")

if len(results['detections_2d']['boxes']) > 0:
    print(f"\n   Detected objects:")
    classes = results['detections_2d']['classes']
    scores = results['detections_2d']['scores']
    for i, (cls, score) in enumerate(zip(classes, scores)):
        class_name = pipeline.config['yolo']['classes'][int(cls)] if int(cls) < len(pipeline.config['yolo']['classes']) else 'unknown'
        print(f"     {i+1}. {class_name} (confidence: {score:.2f})")

# Create visualization
print("\n5. Creating visualization...")
try:
    visualization = pipeline.visualize_results(image, results)
    
    # Save results
    output_path = os.path.join(output_dir, 'test_result.jpg')
    cv2.imwrite(output_path, visualization)
    print(f"   ✓ Saved visualization to: {output_path}")
    
    # Save BEV map separately
    bev_path = os.path.join(output_dir, 'test_bev_map.jpg')
    cv2.imwrite(bev_path, results['bev_map'])
    print(f"   ✓ Saved BEV map to: {bev_path}")
    
except Exception as e:
    print(f"   ✗ Error creating visualization: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
print(f"\nCheck outputs in: {output_dir}/")
