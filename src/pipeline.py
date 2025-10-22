"""
Main YOLO-BEV Pipeline
"""

import torch
import numpy as np
import yaml
import os
from pathlib import Path

from src.models.yolo_detector import YOLODetector
from src.models.estimator_3d import Estimator3D
from src.utils.bev_transform import BEVTransform
from src.utils.visualization import Visualizer


class YOLOBEVPipeline:
    """
    End-to-end pipeline for 3D object detection and BEV transformation
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(
            self.config['device'] if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Initialize modules
        print("Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(self.config)
        
        print("Initializing 3D estimator...")
        self.estimator_3d = Estimator3D(self.config).to(self.device)
        
        print("Initializing BEV transform...")
        self.bev_transform = BEVTransform(self.config)
        
        print("Initializing visualizer...")
        self.visualizer = Visualizer(self.config)
        
        print("Pipeline initialized successfully!")
    
    def process_image(self, image, camera_intrinsic=None):
        """
        Process a single image through the complete pipeline
        
        Args:
            image: Input image (numpy array or path)
            camera_intrinsic: Camera intrinsic matrix (optional)
        
        Returns:
            results: Dictionary containing all detection and BEV results
        """
        # Step 1: 2D Object Detection with YOLO
        print("Running 2D object detection...")
        detections_2d = self.yolo_detector.detect(image)
        
        if len(detections_2d['boxes']) == 0:
            print("No objects detected")
            return {
                'detections_2d': detections_2d,
                'boxes_3d': [],
                'boxes_bev': [],
                'bev_map': self.bev_transform.create_bev_map([])
            }
        
        print(f"Detected {len(detections_2d['boxes'])} objects")
        
        # Step 2: Prepare for 3D estimation
        # Convert image to tensor if needed
        if isinstance(image, str):
            import cv2
            image_array = cv2.imread(image)
        else:
            image_array = image
        
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Extract RoIs
        boxes_2d_list = [detections_2d['boxes']]
        roi_images, roi_indices = self.estimator_3d.extract_rois(
            image_tensor, boxes_2d_list
        )
        
        # Step 3: 3D Bounding Box Estimation
        print("Estimating 3D bounding boxes...")
        if len(roi_images) > 0:
            with torch.no_grad():
                predictions_3d = self.estimator_3d(roi_images.to(self.device))
            
            # Convert predictions to 3D boxes
            boxes_3d = self._predictions_to_boxes_3d(
                predictions_3d,
                detections_2d['boxes']
            )
        else:
            boxes_3d = []
        
        print(f"Estimated {len(boxes_3d)} 3D bounding boxes")
        
        # Step 4: Transform to BEV
        print("Transforming to BEV space...")
        boxes_bev = self.bev_transform.camera_to_bev(boxes_3d)
        
        print(f"Projected {len(boxes_bev)} boxes to BEV")
        
        # Step 5: Create BEV visualization
        bev_map = self.bev_transform.create_bev_map(
            boxes_bev,
            detections_2d['classes']
        )
        
        results = {
            'detections_2d': detections_2d,
            'boxes_3d': boxes_3d,
            'boxes_bev': boxes_bev,
            'bev_map': bev_map
        }
        
        return results
    
    def _predictions_to_boxes_3d(self, predictions, boxes_2d):
        """
        Convert network predictions to 3D bounding boxes
        
        Args:
            predictions: Dictionary of predicted 3D parameters
            boxes_2d: 2D bounding boxes
        
        Returns:
            boxes_3d: List of 3D boxes [x, y, z, w, h, l, yaw]
        """
        boxes_3d = []
        
        dimensions = predictions['dimensions'].cpu().numpy()
        orientations = predictions['orientation'].cpu().numpy()
        depths = predictions['depth'].cpu().numpy()
        location_offsets = predictions['location_offset'].cpu().numpy()
        
        for i, box_2d in enumerate(boxes_2d):
            x1, y1, x2, y2 = box_2d
            cx_2d = (x1 + x2) / 2
            cy_2d = (y1 + y2) / 2
            
            # Get 3D parameters
            w, h, l = dimensions[i]
            yaw = orientations[i, 0]
            z = depths[i, 0]  # depth (forward distance)
            
            # Estimate x, y from 2D box center and depth
            # This is a simplified approach - would need camera intrinsics for accurate projection
            x = location_offsets[i, 0]
            y = location_offsets[i, 1]
            
            boxes_3d.append([x, y, z, w, h, l, yaw])
        
        return boxes_3d
    
    def visualize_results(self, image, results, save_path=None):
        """
        Visualize detection results
        
        Args:
            image: Original image
            results: Results from process_image
            save_path: Optional path to save visualization
        
        Returns:
            visualization: Combined visualization image
        """
        # Create combined visualization
        visualization = self.visualizer.create_combined_visualization(
            image if isinstance(image, np.ndarray) else np.array(image),
            results['bev_map'],
            results['detections_2d']
        )
        
        if save_path:
            self.visualizer.save_visualization(visualization, save_path)
        
        return visualization
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.estimator_3d.load_state_dict(checkpoint['estimator_3d'])
        print("Checkpoint loaded successfully")
    
    def save_checkpoint(self, checkpoint_path, epoch=None, optimizer=None):
        """Save model checkpoint"""
        checkpoint = {
            'estimator_3d': self.estimator_3d.state_dict(),
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
