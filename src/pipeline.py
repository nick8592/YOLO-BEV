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
        
        print(f"Projected {len(boxes_bev)} boxes to BEV (from {len(boxes_3d)} 3D boxes)")
        
        # Debug: Print sample 3D boxes to understand depth distribution
        if len(boxes_3d) > 0:
            depths = [box[2] for box in boxes_3d]
            print(f"Depth range: min={min(depths):.2f}m, max={max(depths):.2f}m, avg={np.mean(depths):.2f}m")
            
            sample_box = boxes_3d[0]
            print(f"Sample 3D box: x={sample_box[0]:.2f}m, y={sample_box[1]:.2f}m, z={sample_box[2]:.2f}m")
            if len(boxes_bev) > 0:
                sample_bev = boxes_bev[0]
                print(f"Sample BEV box: x={sample_bev[0]:.2f}, y={sample_bev[1]:.2f}")
            
            # Show how many boxes were filtered
            if len(boxes_bev) < len(boxes_3d):
                print(f"⚠️  {len(boxes_3d) - len(boxes_bev)} boxes filtered (outside BEV range)")
        
        # Step 5: Create BEV visualization
        bev_map = self.bev_transform.create_bev_map(
            boxes_bev,
            detections_2d['classes'] if len(boxes_bev) > 0 else None
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
        orientations = predictions['orientation'].cpu().numpy()  # [N, 1] - already converted to angle
        depths = predictions['depth'].cpu().numpy()
        location_offsets = predictions['location_offset'].cpu().numpy()
        
        # Image dimensions for normalized coordinates
        img_width = self.config['image']['input_size'][1]
        img_height = self.config['image']['input_size'][0]
        
        for i, box_2d in enumerate(boxes_2d):
            x1, y1, x2, y2 = box_2d
            cx_2d = (x1 + x2) / 2
            cy_2d = (y1 + y2) / 2
            box_height = y2 - y1
            
            # Get 3D parameters
            w, h, l = np.abs(dimensions[i])  # Ensure positive dimensions
            w = max(w, 0.5)  # Minimum 0.5m width
            h = max(h, 0.5)  # Minimum 0.5m height
            l = max(l, 0.5)  # Minimum 0.5m length
            
            # Get orientation (already converted to yaw angle in model forward pass)
            yaw = orientations[i, 0]
            
            # Improved depth estimation using box height and vertical position
            # Objects lower in the image are typically closer
            # Objects with larger bounding boxes are typically closer
            y_normalized = cy_2d / img_height  # 0 (top) to 1 (bottom)
            box_area = (x2 - x1) * (y2 - y1)
            relative_size = box_area / (img_width * img_height)
            
            # Base depth from network prediction (scale it appropriately)
            depth_pred = np.abs(depths[i, 0])
            
            # Scale depth: combine network prediction with heuristics
            # Closer objects (bottom of image) should have smaller z
            # Larger objects should have smaller z
            z_base = 10.0 + depth_pred * 30.0  # Scale to 10-40m range
            z_position = (1.0 - y_normalized) * 20.0  # Objects at top: +20m, bottom: 0m
            z_size = (1.0 - np.clip(relative_size * 50, 0, 0.8)) * 15.0  # Large boxes: closer
            
            z = z_base + z_position + z_size
            z = np.clip(z, 5.0, 100.0)  # Clip to reasonable range
            
            # Estimate lateral position (x) from 2D box center
            # Normalize to image center and scale by depth
            x_normalized = (cx_2d - img_width / 2) / (img_width / 2)
            x = x_normalized * z * 0.6  # Lateral offset proportional to depth
            
            # y is height above ground (assume objects on ground plane)
            y = -h / 2  # Center at ground level
            
            # Add location offsets from network (scaled down)
            x += location_offsets[i, 0] * 5.0
            
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
