#!/usr/bin/env python3
"""
Visualize nuScenes dataset samples with 3D annotations.

Usage:
    python tools/visualize_dataset.py --dataroot nuscenes --version v1.0-mini --samples 10
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.data.nuscenes_dataset import NuScenesDataset
from src.utils.visualization import Visualizer
from src.utils.config import load_config


def visualize_samples(dataroot, version, num_samples=10, output_dir='outputs/dataset_viz'):
    """Visualize random samples from dataset."""
    print(f"Loading nuScenes dataset: {version}")
    
    # Load default config
    config = load_config('configs/config.yaml')
    
    # Initialize dataset
    dataset = NuScenesDataset(
        dataroot=dataroot,
        version=version,
        split='train',
        config=config
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    print(f"\nVisualizing {len(indices)} samples...")
    
    for i, idx in enumerate(indices):
        print(f"Processing sample {i+1}/{len(indices)}: index {idx}")
        
        # Get sample
        sample = dataset[idx]
        image = sample['image']
        annotations = sample['annotations']
        
        # Convert image to uint8 if needed
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        # Draw 2D bounding boxes
        image_with_boxes = image.copy()
        for ann in annotations:
            bbox = ann['bbox_2d']
            class_name = ann['class_name']
            
            # Draw box
            cv2.rectangle(image_with_boxes, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(image_with_boxes, class_name,
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visualization
        output_path = os.path.join(output_dir, f'sample_{idx:04d}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        
        # Print annotation info
        print(f"  - Annotations: {len(annotations)}")
        print(f"  - Classes: {[ann['class_name'] for ann in annotations]}")
        print(f"  - Saved to: {output_path}")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Visualize nuScenes dataset samples')
    parser.add_argument('--dataroot', type=str, default='nuscenes',
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='Dataset version')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='outputs/dataset_viz',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_samples(args.dataroot, args.version, args.samples, args.output)


if __name__ == '__main__':
    main()
