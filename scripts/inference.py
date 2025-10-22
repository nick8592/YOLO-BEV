#!/usr/bin/env python
"""
Inference script for YOLO-BEV Pipeline
Run inference on a single image or directory of images
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import YOLOBEVPipeline


def main():
    parser = argparse.ArgumentParser(description='YOLO-BEV Inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--output', type=str, default='outputs/inference',
                       help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Display results in window')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize pipeline
    print("Initializing YOLO-BEV pipeline...")
    pipeline = YOLOBEVPipeline(config_path=args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        pipeline.load_checkpoint(args.checkpoint)
    
    # Process image(s)
    if os.path.isfile(args.image):
        # Single image
        process_single_image(pipeline, args.image, args.output, args.show)
    elif os.path.isdir(args.image):
        # Directory of images
        process_directory(pipeline, args.image, args.output, args.show)
    else:
        print(f"Error: {args.image} is not a valid file or directory")
        return
    
    print("Inference complete!")


def process_single_image(pipeline, image_path, output_dir, show=False):
    """Process a single image"""
    print(f"\nProcessing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run pipeline
    results = pipeline.process_image(image)
    
    # Visualize
    visualization = pipeline.visualize_results(image, results)
    
    # Save
    output_path = os.path.join(
        output_dir,
        f"{Path(image_path).stem}_result.jpg"
    )
    cv2.imwrite(output_path, visualization)
    print(f"Saved visualization to: {output_path}")
    
    # Display if requested
    if show:
        cv2.imshow('YOLO-BEV Results', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print statistics
    print(f"  - 2D detections: {len(results['detections_2d']['boxes'])}")
    print(f"  - 3D boxes: {len(results['boxes_3d'])}")
    print(f"  - BEV boxes: {len(results['boxes_bev'])}")


def process_directory(pipeline, image_dir, output_dir, show=False):
    """Process all images in a directory"""
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for image_path in image_files:
        process_single_image(pipeline, str(image_path), output_dir, show)


if __name__ == '__main__':
    main()
