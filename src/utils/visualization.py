"""
Visualization utilities for YOLO-BEV Pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import torch


class Visualizer:
    """
    Visualization tools for 2D detection, 3D boxes, and BEV representation
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.classes = config['yolo']['classes']
        
        # COCO class names (YOLO uses COCO dataset)
        self.coco_class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
            12: 'parking meter', 13: 'bench', 16: 'bird', 17: 'cat', 18: 'dog'
        }
        
        # Define colors for each class (BGR format for OpenCV)
        self.colors = {
            'car': (255, 0, 0),           # Blue in BGR
            'truck': (255, 128, 0),       # Orange in BGR
            'bus': (255, 255, 0),         # Cyan in BGR
            'trailer': (0, 255, 0),       # Green in BGR
            'construction_vehicle': (0, 255, 255),  # Yellow in BGR
            'pedestrian': (0, 0, 255),    # Red in BGR
            'person': (0, 0, 255),        # Red in BGR (pedestrian)
            'motorcycle': (255, 0, 255),  # Magenta in BGR
            'bicycle': (128, 0, 255),     # Purple in BGR
            'traffic_cone': (255, 255, 255),  # White in BGR
            'traffic light': (0, 200, 200),   # Yellow in BGR
            'barrier': (128, 128, 128),   # Gray in BGR
            'stop sign': (100, 100, 255), # Light red in BGR
        }
    
    def draw_2d_boxes(self, image, detections, scores=None, classes=None):
        """
        Draw 2D bounding boxes on image
        
        Args:
            image: Input image (numpy array)
            detections: 2D bounding boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            classes: Class IDs [N] (COCO class IDs from YOLO)
        
        Returns:
            image: Image with drawn boxes
        """
        image = image.copy()
        
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class and color
            if classes is not None and i < len(classes):
                class_id = int(classes[i])
                # Use COCO class names since YOLO outputs COCO class IDs
                class_name = self.coco_class_names.get(class_id, f'class_{class_id}')
                color = self.colors.get(class_name, (255, 255, 255))
            else:
                class_name = 'object'
                color = (0, 255, 0)
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = class_name
            if scores is not None and i < len(scores):
                label += f' {scores[i]:.2f}'
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(image, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def draw_3d_boxes(self, image, boxes_3d, camera_intrinsic):
        """
        Draw 3D bounding boxes projected on image
        
        Args:
            image: Input image (numpy array)
            boxes_3d: 3D boxes [N, 7] (x, y, z, w, h, l, yaw)
            camera_intrinsic: Camera intrinsic matrix [3, 3]
        
        Returns:
            image: Image with drawn 3D boxes
        """
        image = image.copy()
        
        for box in boxes_3d:
            x, y, z, w, h, l, yaw = box
            
            # Get 3D box corners
            corners_3d = self._get_3d_box_corners(x, y, z, w, h, l, yaw)
            
            # Project to 2D
            corners_2d = self._project_3d_to_2d(corners_3d, camera_intrinsic)
            
            if corners_2d is None:
                continue
            
            # Draw box edges
            self._draw_3d_box_edges(image, corners_2d)
        
        return image
    
    def _get_3d_box_corners(self, x, y, z, w, h, l, yaw):
        """
        Get 8 corners of 3D bounding box in camera coordinates
        
        IMPORTANT: Expects dimensions (w, h, l) to be the actual extents in camera frame:
        - w: width (x-axis extent, lateral)
        - h: height (y-axis extent, vertical)  
        - l: length (z-axis extent, depth)
        
        These should already account for the box orientation, or yaw should be small
        if dimensions are axis-aligned in camera frame.
        """
        # Define box in local coordinates (centered at origin)
        corners = np.array([
            [-w/2, -h/2, -l/2],
            [w/2, -h/2, -l/2],
            [w/2, -h/2, l/2],
            [-w/2, -h/2, l/2],
            [-w/2, h/2, -l/2],
            [w/2, h/2, -l/2],
            [w/2, h/2, l/2],
            [-w/2, h/2, l/2]
        ])
        
        # Rotation matrix (around y-axis for yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot_matrix = np.array([
            [cos_yaw, 0, sin_yaw],
            [0, 1, 0],
            [-sin_yaw, 0, cos_yaw]
        ])
        
        # Rotate and translate
        corners = corners @ rot_matrix.T
        corners[:, 0] += x
        corners[:, 1] += y
        corners[:, 2] += z
        
        return corners
    
    def _project_3d_to_2d(self, points_3d, camera_intrinsic):
        """Project 3D points to 2D image plane"""
        # Filter points behind camera
        if np.any(points_3d[:, 2] < 0):
            return None
        
        # Project
        points_2d = camera_intrinsic @ points_3d.T
        points_2d = points_2d[:2, :] / points_2d[2, :]
        
        return points_2d.T.astype(int)
    
    def _draw_3d_box_edges(self, image, corners_2d, color=(0, 255, 0), thickness=2):
        """Draw edges of 3D bounding box"""
        # Define edges to draw
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for start, end in edges:
            pt1 = tuple(corners_2d[start])
            pt2 = tuple(corners_2d[end])
            cv2.line(image, pt1, pt2, color, thickness)
    
    def visualize_bev(self, bev_map, title="Bird's Eye View"):
        """
        Display BEV map using matplotlib
        
        Args:
            bev_map: BEV occupancy map (numpy array)
            title: Plot title
        """
        plt.figure(figsize=(10, 12))
        plt.imshow(cv2.cvtColor(bev_map, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.xlabel('X (lateral)')
        plt.ylabel('Y (longitudinal)')
        
        # Add axis labels with distances
        x_range = self.config['bev']['x_range']
        y_range = self.config['bev']['y_range']
        plt.xticks(
            np.linspace(0, bev_map.shape[1], 5),
            [f'{x:.0f}m' for x in np.linspace(x_range[0], x_range[1], 5)]
        )
        plt.yticks(
            np.linspace(0, bev_map.shape[0], 5),
            [f'{y:.0f}m' for y in np.linspace(y_range[0], y_range[1], 5)]
        )
        
        plt.tight_layout()
        plt.show()
    
    def create_combined_visualization(self, image, bev_map, detections_2d=None):
        """
        Create side-by-side visualization of camera view and BEV
        
        Args:
            image: Front camera image
            bev_map: BEV map
            detections_2d: Optional 2D detections to draw
        
        Returns:
            combined: Combined visualization image
        """
        # Draw 2D boxes if provided
        if detections_2d is not None:
            image = self.draw_2d_boxes(
                image,
                detections_2d.get('boxes', []),
                detections_2d.get('scores', None),
                detections_2d.get('classes', None)
            )
        
        # Resize images to same height
        target_height = 600
        image_resized = cv2.resize(image, 
                                   (int(image.shape[1] * target_height / image.shape[0]), 
                                    target_height))
        bev_resized = cv2.resize(bev_map, 
                                (int(bev_map.shape[1] * target_height / bev_map.shape[0]), 
                                 target_height))
        
        # Concatenate horizontally
        combined = np.hstack([image_resized, bev_resized])
        
        # Add titles
        cv2.putText(combined, 'Front Camera View', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Bird's Eye View", 
                   (image_resized.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    def save_visualization(self, image, path):
        """Save visualization to file"""
        cv2.imwrite(path, image)
        print(f"Visualization saved to {path}")
    
    def create_bev_comparison(self, bev_pred, bev_gt, detections_2d=None):
        """
        Create side-by-side comparison of predicted and ground truth BEV
        
        Args:
            bev_pred: Predicted BEV map
            bev_gt: Ground truth BEV map
            detections_2d: Optional detection info for title
        
        Returns:
            combined: Side-by-side BEV comparison image
        """
        target_height = 600
        
        # Resize both BEV maps
        bev_pred_resized = cv2.resize(bev_pred, 
                                     (int(bev_pred.shape[1] * target_height / bev_pred.shape[0]), 
                                      target_height))
        bev_gt_resized = cv2.resize(bev_gt, 
                                   (int(bev_gt.shape[1] * target_height / bev_gt.shape[0]), 
                                    target_height))
        
        # Concatenate horizontally
        combined = np.hstack([bev_pred_resized, bev_gt_resized])
        
        # Add titles
        cv2.putText(combined, 'Predicted BEV', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'Ground Truth BEV', 
                   (bev_pred_resized.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add detection count if available
        if detections_2d is not None:
            num_det = len(detections_2d.get('boxes', []))
            cv2.putText(combined, f'Detected: {num_det} objects', 
                       (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined
