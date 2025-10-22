"""
3D Bounding Box Estimation Module
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class Estimator3D(nn.Module):
    """
    3D bounding box estimator using backbone feature extraction
    and regression heads for 3D parameters
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        super(Estimator3D, self).__init__()
        
        self.config = config
        self.backbone_name = config['estimation_3d']['backbone']
        self.feature_dim = config['estimation_3d']['feature_dim']
        self.hidden_dim = config['estimation_3d']['hidden_dim']
        
        # Build backbone
        self.backbone = self._build_backbone()
        
        # Freeze backbone layers (optional, can fine-tune later)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Build regression heads
        self.dimension_head = self._build_head(3)  # width, height, length
        self.orientation_head = self._build_head(2)  # sin(yaw), cos(yaw)
        self.depth_head = self._build_head(1)  # depth (z-coordinate)
        self.location_head = self._build_head(2)  # x, y offset from center
        
    def _build_backbone(self):
        """Build feature extraction backbone"""
        if 'resnet50' in self.backbone_name:
            backbone = models.resnet50(pretrained=True)
            # Remove final FC layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif 'resnet101' in self.backbone_name:
            backbone = models.resnet101(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif 'vgg16' in self.backbone_name:
            backbone = models.vgg16(pretrained=True).features
            self.feature_dim = 512
            
        elif 'vgg19' in self.backbone_name:
            backbone = models.vgg19(pretrained=True).features
            self.feature_dim = 512
            
        elif 'efficientnet' in self.backbone_name:
            backbone = models.efficientnet_b0(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 1280
            
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        return backbone
    
    def _build_head(self, output_dim):
        """Build regression head"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, output_dim)
        )
    
    def forward(self, roi_images):
        """
        Forward pass for 3D estimation
        
        Args:
            roi_images: Batch of RoI images [B, C, H, W]
            
        Returns:
            predictions: Dictionary containing 3D parameters
        """
        # Extract features
        features = self.backbone(roi_images)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Apply regression heads
        dimensions = self.dimension_head(features)  # [B, 3]
        orientation_encoding = self.orientation_head(features)  # [B, 2]
        depth = self.depth_head(features)  # [B, 1]
        location_offset = self.location_head(features)  # [B, 2]
        
        # Convert orientation encoding to angle
        orientation = torch.atan2(
            orientation_encoding[:, 0],
            orientation_encoding[:, 1]
        ).unsqueeze(1)
        
        predictions = {
            'dimensions': dimensions,  # width, height, length
            'orientation': orientation,  # yaw angle
            'depth': depth,
            'location_offset': location_offset
        }
        
        return predictions
    
    def extract_rois(self, images, boxes_2d, target_size=(224, 224)):
        """
        Extract RoI regions from images based on 2D bounding boxes
        
        Args:
            images: Batch of images [B, C, H, W]
            boxes_2d: List of 2D bounding boxes for each image
            target_size: Target size for RoI images
            
        Returns:
            roi_images: Tensor of RoI images
            roi_indices: Indices mapping RoIs to original images
        """
        import torch.nn.functional as F
        
        roi_images = []
        roi_indices = []
        
        for batch_idx, boxes in enumerate(boxes_2d):
            image = images[batch_idx]
            
            for box in boxes:
                x1, y1, x2, y2 = box
                
                # Crop RoI
                roi = image[:, int(y1):int(y2), int(x1):int(x2)]
                
                # Resize to target size
                roi = F.interpolate(
                    roi.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                
                roi_images.append(roi.squeeze(0))
                roi_indices.append(batch_idx)
        
        if len(roi_images) > 0:
            roi_images = torch.stack(roi_images)
            roi_indices = torch.LongTensor(roi_indices)
        else:
            roi_images = torch.empty(0, images.size(1), *target_size)
            roi_indices = torch.empty(0, dtype=torch.long)
        
        return roi_images, roi_indices


class Loss3D(nn.Module):
    """
    Loss function for 3D bounding box estimation
    """
    
    def __init__(self, config):
        super(Loss3D, self).__init__()
        
        self.weights = config['training']['loss_weights']
        
        self.dimension_loss = nn.SmoothL1Loss()
        self.orientation_loss = nn.SmoothL1Loss()
        self.depth_loss = nn.SmoothL1Loss()
        self.location_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
        """
        Compute 3D estimation loss
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of ground truth values
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        loss_dim = self.dimension_loss(predictions['dimensions'], targets['dimensions'])
        loss_orient = self.orientation_loss(predictions['orientation'], targets['orientation'])
        loss_depth = self.depth_loss(predictions['depth'], targets['depth'])
        loss_loc = self.location_loss(predictions['location_offset'], targets['location_offset'])
        
        # Weighted sum
        total_loss = (
            self.weights['box_3d'] * loss_dim +
            self.weights['orientation'] * loss_orient +
            self.weights['depth'] * loss_depth +
            loss_loc
        )
        
        loss_dict = {
            'loss_3d_total': total_loss.item(),
            'loss_dimensions': loss_dim.item(),
            'loss_orientation': loss_orient.item(),
            'loss_depth': loss_depth.item(),
            'loss_location': loss_loc.item()
        }
        
        return total_loss, loss_dict
