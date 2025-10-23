#!/usr/bin/env python3
"""
Pre-Training Readiness Check for YOLO-BEV
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("YOLO-BEV Training Readiness Check")
print("=" * 70)

issues = []
warnings = []

# 1. Check Python environment
print("\n1. Environment Check")
print("-" * 70)
try:
    import torch
    import torchvision
    import numpy as np
    import cv2
    from ultralytics import YOLO
    from nuscenes.nuscenes import NuScenes
    print(f"   ✓ Python {sys.version.split()[0]}")
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        warnings.append("CUDA not available - training will be VERY slow on CPU")
except Exception as e:
    issues.append(f"Missing dependencies: {e}")

# 2. Check configuration
print("\n2. Configuration Check")
print("-" * 70)
try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ Configuration loaded")
    print(f"   - Dataset: {config['dataset']['name']}")
    print(f"   - YOLO model: {config['yolo']['model_name']}")
    print(f"   - 3D backbone: {config['estimation_3d']['backbone']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Epochs: {config['training']['num_epochs']}")
    print(f"   - Learning rate: {config['training']['learning_rate']}")
except Exception as e:
    issues.append(f"Configuration error: {e}")
    config = None

# 3. Check nuScenes dataset
print("\n3. Dataset Check")
print("-" * 70)
if config:
    try:
        from src.data.nuscenes_dataset import NuScenesDataset
        
        # Check if dataset path exists
        dataset_path = config['dataset']['data_root']
        if not os.path.exists(dataset_path):
            issues.append(f"Dataset not found at: {dataset_path}")
        else:
            print(f"   ✓ Dataset path exists: {dataset_path}")
            
            # Try to load dataset
            print("   Loading dataset (this may take a moment)...")
            dataset = NuScenesDataset(config, split='train')
            print(f"   ✓ Dataset loaded: {len(dataset)} samples")
            
            # Test loading a sample
            sample = dataset[0]
            print(f"   ✓ Sample loaded successfully")
            print(f"   - Image shape: {sample['image'].shape}")
            print(f"   - Annotations available: {len(sample['annotations']['boxes_2d'])}")
            
            if len(dataset) < 10:
                warnings.append(f"Very few samples ({len(dataset)}) - consider using full dataset")
                
    except Exception as e:
        issues.append(f"Dataset loading error: {e}")
        import traceback
        traceback.print_exc()

# 4. Check model initialization
print("\n4. Model Initialization Check")
print("-" * 70)
if config:
    try:
        from src.models.yolo_detector import YOLODetector
        from src.models.estimator_3d import Estimator3D, Loss3D
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize YOLO
        print("   Initializing YOLO detector...")
        yolo = YOLODetector(config)
        print(f"   ✓ YOLO initialized ({config['yolo']['model_name']})")
        
        # Initialize 3D estimator
        print("   Initializing 3D estimator...")
        estimator = Estimator3D(config).to(device)
        print(f"   ✓ 3D Estimator initialized ({config['estimation_3d']['backbone']})")
        
        # Count parameters
        total_params = sum(p.numel() for p in estimator.parameters())
        trainable_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Initialize loss
        criterion = Loss3D(config)
        print(f"   ✓ Loss function initialized")
        
        # Test forward pass
        print("   Testing forward pass...")
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = estimator(dummy_input)
        print(f"   ✓ Forward pass successful")
        
    except Exception as e:
        issues.append(f"Model initialization error: {e}")
        import traceback
        traceback.print_exc()

# 5. Check output directories
print("\n5. Output Directories Check")
print("-" * 70)
if config:
    dirs_to_check = [
        config['paths']['checkpoint_dir'],
        config['paths']['output_dir'],
        config['paths']['log_dir']
    ]
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"   ✓ {dir_path} exists")
        else:
            print(f"   Creating {dir_path}...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✓ {dir_path} created")

# 6. Check disk space
print("\n6. Disk Space Check")
print("-" * 70)
try:
    import shutil
    stat = shutil.disk_usage('/')
    free_gb = stat.free / (1024**3)
    print(f"   Available disk space: {free_gb:.1f} GB")
    if free_gb < 5:
        warnings.append(f"Low disk space: {free_gb:.1f} GB (recommended: >10 GB)")
    else:
        print(f"   ✓ Sufficient disk space")
except Exception as e:
    warnings.append(f"Could not check disk space: {e}")

# 7. Check GPU memory
print("\n7. GPU Memory Check")
print("-" * 70)
if torch.cuda.is_available():
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / (1024**3)
        print(f"   Total GPU memory: {total_memory:.1f} GB")
        
        # Check available memory
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free = total_memory - reserved
        
        print(f"   Available GPU memory: {free:.1f} GB")
        
        if total_memory < 4:
            warnings.append(f"Limited GPU memory: {total_memory:.1f} GB - consider reducing batch size")
        else:
            print(f"   ✓ Sufficient GPU memory")
            
    except Exception as e:
        warnings.append(f"Could not check GPU memory: {e}")
else:
    warnings.append("No GPU available - training will be slow")

# 8. Estimate training time
print("\n8. Training Time Estimate")
print("-" * 70)
if config and 'dataset' in locals():
    num_samples = len(dataset)
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    
    iterations_per_epoch = num_samples // batch_size
    total_iterations = iterations_per_epoch * num_epochs
    
    # Rough estimate: 0.5 seconds per iteration on GPU, 5 seconds on CPU
    time_per_iter = 5.0 if not torch.cuda.is_available() else 0.5
    estimated_hours = (total_iterations * time_per_iter) / 3600
    
    print(f"   Dataset samples: {num_samples}")
    print(f"   Batch size: {batch_size}")
    print(f"   Iterations per epoch: {iterations_per_epoch}")
    print(f"   Total epochs: {num_epochs}")
    print(f"   Total iterations: {total_iterations:,}")
    print(f"   Estimated training time: {estimated_hours:.1f} hours")
    
    if estimated_hours > 48:
        warnings.append(f"Training will take ~{estimated_hours:.0f} hours - consider reducing epochs or using GPU")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if issues:
    print("\n❌ CRITICAL ISSUES (must fix before training):")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")

if warnings:
    print("\n⚠️  WARNINGS (recommended to address):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

if not issues and not warnings:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nThe system is ready for training.")
    print("\nTo start training, run:")
    print("   python3 scripts/train.py --config configs/config.yaml")
    print("\nOptional arguments:")
    print("   --epochs N        - Override number of epochs")
    print("   --resume path.pth - Resume from checkpoint")
elif not issues:
    print("\n⚠️  TRAINING IS POSSIBLE but with warnings above.")
    print("\nTo start training anyway, run:")
    print("   python3 scripts/train.py --config configs/config.yaml")
else:
    print("\n❌ TRAINING NOT READY - Please fix the critical issues above.")

print("\n" + "=" * 70)
