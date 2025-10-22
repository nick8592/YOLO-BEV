"""
YOLO-based 2D Object Detection Module
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np


class YOLODetector(nn.Module):
    """
    YOLO-based 2D object detector for nuScenes front camera images
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        super(YOLODetector, self).__init__()
        
        self.config = config
        self.model_name = config['yolo']['model_name']
        self.conf_threshold = config['yolo']['confidence_threshold']
        self.iou_threshold = config['yolo']['iou_threshold']
        self.classes = config['yolo']['classes']
        
        # Load YOLO model
        self.model = YOLO(f"{self.model_name}.pt")
        
        # Set device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def forward(self, images):
        """
        Forward pass for detection
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            detections: List of detection results for each image
        """
        results = self.model(images, conf=self.conf_threshold, iou=self.iou_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            detection = {
                'boxes': boxes.xyxy.cpu().numpy(),  # [N, 4] (x1, y1, x2, y2)
                'scores': boxes.conf.cpu().numpy(),  # [N]
                'classes': boxes.cls.cpu().numpy(),  # [N]
            }
            
            detections.append(detection)
        
        return detections
    
    def detect(self, image):
        """
        Detect objects in a single image
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            detection: Dictionary containing boxes, scores, and classes
        """
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        result = results[0]
        boxes = result.boxes
        
        detection = {
            'boxes': boxes.xyxy.cpu().numpy(),
            'scores': boxes.conf.cpu().numpy(),
            'classes': boxes.cls.cpu().numpy(),
        }
        
        return detection
    
    def train_model(self, data_yaml, epochs=100, imgsz=640, batch=16):
        """
        Train YOLO model on custom dataset
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device
        )
        
        return results
    
    def save(self, path):
        """Save model weights"""
        self.model.save(path)
    
    def load(self, path):
        """Load model weights"""
        self.model = YOLO(path)
        self.model.to(self.device)
