#!/usr/bin/env python
"""
Demo script for YOLO-BEV Pipeline
Quick demonstration on nuScenes sample data
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import YOLOBEVPipeline


def main():
    print("="*60)
    print("YOLO-BEV Pipeline Demo")
    print("="*60)
    
    # Configuration
    config_path = 'configs/config.yaml'
    output_dir = 'outputs/demo'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = YOLOBEVPipeline(config_path=config_path)
    
    # Find a sample image from nuScenes
    print("\n2. Loading sample image from nuScenes...")
    nuscenes_path = 'nuscenes/samples/CAM_FRONT'
    
    if not os.path.exists(nuscenes_path):
        print(f"Error: nuScenes data not found at {nuscenes_path}")
        print("Please ensure the nuScenes dataset is properly linked.")
        return
    
    # Get first image
    image_files = list(Path(nuscenes_path).glob('*.jpg'))
    if not image_files:
        print(f"No images found in {nuscenes_path}")
        return
    
    sample_image_path = str(image_files[0])
    print(f"Using sample image: {sample_image_path}")
    
    # Load image
    image = cv2.imread(sample_image_path)
    if image is None:
        print(f"Error: Could not load image")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Process through pipeline
    print("\n3. Processing through YOLO-BEV pipeline...")
    print("   - Running 2D object detection...")
    print("   - Estimating 3D bounding boxes...")
    print("   - Transforming to BEV space...")
    
    results = pipeline.process_image(image)
    
    # Print results
    print("\n4. Results:")
    print(f"   - 2D Detections: {len(results['detections_2d']['boxes'])} objects")
    print(f"   - 3D Boxes: {len(results['boxes_3d'])} objects")
    print(f"   - BEV Boxes: {len(results['boxes_bev'])} objects")
    
    # Create visualization
    print("\n5. Creating visualization...")
    visualization = pipeline.visualize_results(image, results)
    
    # Save results
    output_path = os.path.join(output_dir, 'demo_result.jpg')
    cv2.imwrite(output_path, visualization)
    print(f"   Saved visualization to: {output_path}")
    
    # Save BEV map separately
    bev_path = os.path.join(output_dir, 'bev_map.jpg')
    cv2.imwrite(bev_path, results['bev_map'])
    print(f"   Saved BEV map to: {bev_path}")
    
    # Display results
    print("\n6. Displaying results...")
    print("   Press any key to close the window")
    
    cv2.imshow('YOLO-BEV Demo Results', visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == '__main__':
    main()
