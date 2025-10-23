#!/usr/bin/env python
"""
Training script for YOLO-BEV 3D Estimator
"""

import argparse
import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.nuscenes_dataset import NuScenesDataset
from src.models.estimator_3d import Estimator3D, Loss3D
from src.models.yolo_detector import YOLODetector


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-BEV 3D Estimator')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Initialize dataset and dataloader
    print("Loading dataset...")
    train_dataset = NuScenesDataset(config, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Initialize models
    print("Initializing models...")
    yolo_detector = YOLODetector(config)
    estimator_3d = Estimator3D(config).to(device)
    
    # Initialize loss and optimizer
    criterion = Loss3D(config)
    optimizer = optim.Adam(
        estimator_3d.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        estimator_3d.load_state_dict(checkpoint['estimator_3d'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Training loop
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train one epoch
        train_loss = train_epoch(
            estimator_3d,
            yolo_detector,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch
        )
        
        # Update learning rate (after optimizer.step() has been called in train_epoch)
        scheduler.step()
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} - Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'estimator_3d': estimator_3d.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.pth')
    torch.save({
        'estimator_3d': estimator_3d.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


def train_epoch(model, yolo_detector, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Training")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        annotations = batch['annotations']
        
        # Skip if no annotations
        if len(annotations['boxes_2d']) == 0:
            continue
        
        # Extract RoIs using ground truth 2D boxes
        boxes_2d_list = [ann.cpu().numpy() for ann in annotations['boxes_2d']]
        roi_images, roi_indices = model.extract_rois(images, boxes_2d_list)
        
        if len(roi_images) == 0:
            continue
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(roi_images.to(device))
        
        # Prepare targets
        targets = prepare_targets(annotations, roi_indices, device)
        
        # Compute loss
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def prepare_targets(annotations, roi_indices, device):
    """Prepare ground truth targets for training"""
    targets = {
        'dimensions': [],
        'orientation': [],
        'depth': [],
        'location_offset': []
    }
    
    for idx in roi_indices:
        # Get annotations for this image
        boxes_3d = annotations['boxes_3d'][idx]  # [x, y, z, w, h, l] in camera coords
        orientations = annotations['orientations'][idx]
        depths = annotations['depths'][idx]
        
        # Extract target values
        if len(boxes_3d) > 0:
            targets['dimensions'].append(boxes_3d[3:6])  # w, h, l (indices 3, 4, 5)
            targets['orientation'].append(orientations)
            targets['depth'].append(depths)
            targets['location_offset'].append([boxes_3d[0], boxes_3d[1]])  # x, y offsets
    
    # Convert to tensors
    for key in targets:
        if len(targets[key]) > 0:
            targets[key] = torch.FloatTensor(targets[key]).to(device)
        else:
            targets[key] = torch.FloatTensor([]).to(device)
    
    return targets


if __name__ == '__main__':
    main()
