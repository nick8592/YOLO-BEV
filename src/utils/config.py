"""
Configuration utilities
"""

import yaml
import os
from pathlib import Path


def load_config(config_path='configs/config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to {config_path}")


def validate_config(config):
    """
    Validate configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        valid: True if configuration is valid
        errors: List of validation errors
    """
    errors = []
    
    # Check required keys
    required_keys = ['dataset', 'yolo', 'estimation_3d', 'bev', 'training']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")
    
    # Check dataset path
    if 'dataset' in config:
        data_root = config['dataset'].get('data_root')
        if data_root and not os.path.exists(data_root):
            errors.append(f"Dataset path does not exist: {data_root}")
    
    # Check BEV ranges
    if 'bev' in config:
        x_range = config['bev'].get('x_range', [-50, 50])
        y_range = config['bev'].get('y_range', [0, 100])
        resolution = config['bev'].get('resolution', 0.2)
        
        if x_range[0] >= x_range[1]:
            errors.append("BEV x_range[0] must be less than x_range[1]")
        if y_range[0] >= y_range[1]:
            errors.append("BEV y_range[0] must be less than y_range[1]")
        if resolution <= 0:
            errors.append("BEV resolution must be positive")
    
    valid = len(errors) == 0
    return valid, errors


def get_default_config():
    """
    Get default configuration
    
    Returns:
        config: Default configuration dictionary
    """
    config = {
        'dataset': {
            'name': 'nuscenes',
            'version': 'v1.0-mini',
            'data_root': './nuscenes',
            'split': 'train'
        },
        'image': {
            'height': 900,
            'width': 1600,
            'input_size': [384, 1280],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'yolo': {
            'model_name': 'yolov8n',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'classes': ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                       'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        },
        'estimation_3d': {
            'backbone': 'resnet50',
            'feature_dim': 2048,
            'hidden_dim': 512
        },
        'bev': {
            'x_range': [-50, 50],
            'y_range': [0, 100],
            'resolution': 0.2,
            'z_range': [-10, 10]
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'loss_weights': {
                'cls': 1.0,
                'box_2d': 1.0,
                'box_3d': 2.0,
                'orientation': 1.5,
                'depth': 2.0
            }
        },
        'paths': {
            'checkpoint_dir': './checkpoints',
            'output_dir': './outputs',
            'log_dir': './logs'
        },
        'device': 'cuda',
        'num_workers': 4,
        'seed': 42
    }
    
    return config
