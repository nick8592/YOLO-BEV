#!/usr/bin/env python3
"""
Evaluate trained YOLO-BEV model on validation/test set.

Usage:
    python tools/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from src.pipeline import YOLOBEVPipeline
from src.data.nuscenes_dataset import NuScenesDataset
from src.utils.config import load_config


def calculate_metrics(predictions, ground_truths):
    """Calculate evaluation metrics for 3D object detection."""
    metrics = {
        'mAP': 0.0,
        'dimension_error': [],
        'orientation_error': [],
        'depth_error': [],
        'location_error': []
    }
    
    for pred, gt in zip(predictions, ground_truths):
        if len(gt) == 0:
            continue
            
        # Calculate dimension error (meters)
        if len(pred) > 0:
            dim_error = np.mean(np.abs(pred['dimensions'] - gt['dimensions']))
            metrics['dimension_error'].append(dim_error)
            
            # Calculate orientation error (radians)
            ori_error = np.mean(np.abs(pred['orientation'] - gt['orientation']))
            metrics['orientation_error'].append(ori_error)
            
            # Calculate depth error (meters)
            depth_error = np.mean(np.abs(pred['depth'] - gt['depth']))
            metrics['depth_error'].append(depth_error)
            
            # Calculate 3D location error (meters)
            loc_error = np.mean(np.linalg.norm(pred['location'] - gt['location'], axis=-1))
            metrics['location_error'].append(loc_error)
    
    # Aggregate metrics
    metrics['avg_dimension_error'] = np.mean(metrics['dimension_error']) if metrics['dimension_error'] else 0.0
    metrics['avg_orientation_error'] = np.mean(metrics['orientation_error']) if metrics['orientation_error'] else 0.0
    metrics['avg_depth_error'] = np.mean(metrics['depth_error']) if metrics['depth_error'] else 0.0
    metrics['avg_location_error'] = np.mean(metrics['location_error']) if metrics['location_error'] else 0.0
    
    return metrics


def evaluate(config_path, checkpoint_path, split='val'):
    """Evaluate model on dataset."""
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = YOLOBEVPipeline(config)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        pipeline.load_checkpoint(checkpoint_path)
    else:
        print("Warning: No checkpoint provided or not found!")
        return
    
    # Load dataset
    print(f"Loading {split} dataset...")
    dataset = NuScenesDataset(
        dataroot=config['data']['data_root'],
        version=config['data']['version'],
        split=split,
        config=config
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Evaluation loop
    predictions = []
    ground_truths = []
    
    pipeline.yolo_detector.model.eval()
    pipeline.estimator.eval()
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            # Get sample
            sample = dataset[idx]
            image = sample['image']
            annotations = sample['annotations']
            
            # Run inference
            try:
                results = pipeline.process_image(image)
                predictions.append(results)
                ground_truths.append(annotations)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(predictions, ground_truths)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Split: {split}")
    print(f"Samples evaluated: {len(predictions)}")
    print(f"\nAverage Dimension Error: {metrics['avg_dimension_error']:.3f} m")
    print(f"Average Orientation Error: {metrics['avg_orientation_error']:.3f} rad")
    print(f"Average Depth Error: {metrics['avg_depth_error']:.3f} m")
    print(f"Average 3D Location Error: {metrics['avg_location_error']:.3f} m")
    print("="*50)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO-BEV model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint, args.split)


if __name__ == '__main__':
    main()
