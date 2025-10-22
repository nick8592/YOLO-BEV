"""
nuScenes Dataset Loader for YOLO-BEV Pipeline
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import cv2


class NuScenesDataset(Dataset):
    """
    nuScenes dataset loader for front camera images with 3D annotations
    """
    
    def __init__(self, config, split='train', transform=None):
        """
        Args:
            config: Configuration dictionary
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Initialize nuScenes
        self.nusc = NuScenes(
            version=config['dataset']['version'],
            dataroot=config['dataset']['data_root'],
            verbose=True
        )
        
        # Get front camera samples
        self.samples = self._get_samples()
        
        # Class mapping
        self.classes = config['yolo']['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def _get_samples(self):
        """Get front camera samples from nuScenes"""
        samples = []
        for scene in self.nusc.scene:
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                # Get front camera data
                cam_front_token = sample['data']['CAM_FRONT']
                samples.append(cam_front_token)
                sample_token = sample['next']
                
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Preprocessed image tensor
            annotations: Dictionary containing 2D and 3D bounding box information
        """
        cam_token = self.samples[idx]
        cam_data = self.nusc.get('sample_data', cam_token)
        
        # Load image
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Get camera calibration
        calibration = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(calibration['camera_intrinsic'])
        
        # Get annotations
        sample = self.nusc.get('sample', cam_data['sample_token'])
        annotations = self._get_annotations(sample, cam_data, camera_intrinsic)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transform(image)
        
        return {
            'image': image,
            'annotations': annotations,
            'camera_intrinsic': torch.FloatTensor(camera_intrinsic),
            'sample_token': cam_data['sample_token']
        }
    
    def _get_annotations(self, sample, cam_data, camera_intrinsic):
        """Extract 2D and 3D bounding box annotations"""
        annotations = {
            'boxes_2d': [],
            'boxes_3d': [],
            'classes': [],
            'orientations': [],
            'depths': []
        }
        
        # Get 3D boxes
        boxes_3d = self.nusc.get_boxes(cam_data['token'])
        
        for box in boxes_3d:
            # Filter by class
            if box.name.split('.')[0] not in self.class_to_idx:
                continue
            
            # Get 2D bounding box
            corners = box.corners()  # 3x8 array
            corners_2d = self._project_to_image(corners, camera_intrinsic)
            
            if corners_2d is None:
                continue
            
            x_min, y_min = np.min(corners_2d, axis=1)
            x_max, y_max = np.max(corners_2d, axis=1)
            
            # Clip to image boundaries
            h, w = self.config['image']['height'], self.config['image']['width']
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Store annotations
            annotations['boxes_2d'].append([x_min, y_min, x_max, y_max])
            annotations['boxes_3d'].append(box.center.tolist() + box.wlh.tolist())
            annotations['classes'].append(self.class_to_idx[box.name.split('.')[0]])
            annotations['orientations'].append(box.orientation.yaw_pitch_roll[0])
            annotations['depths'].append(np.linalg.norm(box.center[:2]))
        
        # Convert to tensors
        for key in annotations:
            if len(annotations[key]) > 0:
                annotations[key] = torch.FloatTensor(annotations[key])
            else:
                annotations[key] = torch.FloatTensor([])
        
        return annotations
    
    def _project_to_image(self, points_3d, camera_intrinsic):
        """Project 3D points to 2D image plane"""
        # Filter points behind camera
        if np.any(points_3d[2, :] < 0):
            return None
        
        # Project to image
        points_2d = camera_intrinsic @ points_3d
        points_2d = points_2d[:2, :] / points_2d[2, :]
        
        return points_2d
    
    def _default_transform(self, image):
        """Default image transformation"""
        # Resize
        target_size = self.config['image']['input_size']
        image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # Convert to tensor and normalize
        image = np.array(image).astype(np.float32) / 255.0
        mean = np.array(self.config['image']['mean'])
        std = np.array(self.config['image']['std'])
        image = (image - mean) / std
        
        # Convert to CHW format
        image = torch.FloatTensor(image).permute(2, 0, 1)
        
        return image


def get_dataloader(config, split='train', batch_size=None, shuffle=True):
    """
    Create a dataloader for nuScenes dataset
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        batch_size: Batch size (defaults to config value)
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    dataset = NuScenesDataset(config, split=split)
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    return dataloader
