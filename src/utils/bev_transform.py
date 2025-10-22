"""
BEV (Bird's Eye View) Transformation Utilities
"""

import numpy as np
import torch
import cv2


class BEVTransform:
    """
    Transform 3D bounding boxes from camera coordinates to BEV space
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # BEV parameters
        self.x_range = config['bev']['x_range']  # [-50, 50] meters
        self.y_range = config['bev']['y_range']  # [0, 100] meters
        self.z_range = config['bev']['z_range']  # [-10, 10] meters
        self.resolution = config['bev']['resolution']  # meters per pixel
        
        # Calculate BEV grid size
        self.bev_width = int((self.x_range[1] - self.x_range[0]) / self.resolution)
        self.bev_height = int((self.y_range[1] - self.y_range[0]) / self.resolution)
        
    def camera_to_bev(self, boxes_3d):
        """
        Transform 3D bounding boxes from camera coordinates to BEV coordinates
        
        Args:
            boxes_3d: List of 3D boxes in camera coordinates
                     Each box: [x, y, z, width, height, length, yaw]
        
        Returns:
            boxes_bev: List of boxes in BEV coordinates
                      Each box: [x_bev, y_bev, width, length, yaw]
        """
        boxes_bev = []
        
        for box in boxes_3d:
            x, y, z, w, h, l, yaw = box
            
            # Filter by z-range (height above ground)
            if z < self.z_range[0] or z > self.z_range[1]:
                continue
            
            # Convert to BEV coordinates
            # In camera coords: x (right), y (down), z (forward)
            # In BEV: x (right), y (forward)
            x_bev = self._world_to_bev_x(x)
            y_bev = self._world_to_bev_y(z)  # Note: z maps to y in BEV
            
            # Check if within BEV range
            if (x_bev < 0 or x_bev >= self.bev_width or 
                y_bev < 0 or y_bev >= self.bev_height):
                continue
            
            # Convert dimensions to BEV pixels
            width_bev = w / self.resolution
            length_bev = l / self.resolution
            
            boxes_bev.append([x_bev, y_bev, width_bev, length_bev, yaw])
        
        return boxes_bev
    
    def _world_to_bev_x(self, x):
        """Convert world x-coordinate to BEV pixel x"""
        return int((x - self.x_range[0]) / self.resolution)
    
    def _world_to_bev_y(self, y):
        """Convert world y-coordinate to BEV pixel y"""
        return int((y - self.y_range[0]) / self.resolution)
    
    def create_bev_map(self, boxes_bev, class_ids=None):
        """
        Create a BEV occupancy map from bounding boxes
        
        Args:
            boxes_bev: List of boxes in BEV coordinates
            class_ids: List of class IDs for each box
        
        Returns:
            bev_map: BEV occupancy map [H, W, 3] (RGB image)
        """
        # Create blank BEV map
        bev_map = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # Define colors for different object classes
        colors = {
            0: (255, 0, 0),      # car - red
            1: (255, 128, 0),    # truck - orange
            2: (255, 255, 0),    # bus - yellow
            3: (0, 255, 0),      # trailer - green
            4: (0, 255, 255),    # construction - cyan
            5: (0, 0, 255),      # pedestrian - blue
            6: (255, 0, 255),    # motorcycle - magenta
            7: (128, 0, 255),    # bicycle - purple
            8: (255, 255, 255),  # traffic_cone - white
            9: (128, 128, 128),  # barrier - gray
        }
        default_color = (200, 200, 200)
        
        # Draw boxes on BEV map
        for i, box in enumerate(boxes_bev):
            x_bev, y_bev, width, length, yaw = box
            
            # Get color
            color = default_color
            if class_ids is not None and i < len(class_ids):
                class_id = int(class_ids[i])
                color = colors.get(class_id, default_color)
            
            # Get rotated box corners
            corners = self._get_box_corners(x_bev, y_bev, width, length, yaw)
            
            # Draw filled polygon
            cv2.fillPoly(bev_map, [corners], color)
            
            # Draw box outline
            cv2.polylines(bev_map, [corners], True, (255, 255, 255), 2)
            
            # Draw orientation arrow
            arrow_length = max(width, length) * 0.5
            end_x = int(x_bev + arrow_length * np.cos(yaw))
            end_y = int(y_bev + arrow_length * np.sin(yaw))
            cv2.arrowedLine(bev_map, (int(x_bev), int(y_bev)), 
                          (end_x, end_y), (0, 255, 0), 2)
        
        # Draw ego vehicle (at the center bottom)
        ego_x = self.bev_width // 2
        ego_y = self.bev_height - 20
        ego_width = int(1.8 / self.resolution)  # 1.8m typical car width
        ego_length = int(4.5 / self.resolution)  # 4.5m typical car length
        
        ego_corners = self._get_box_corners(ego_x, ego_y, ego_width, ego_length, 0)
        cv2.fillPoly(bev_map, [ego_corners], (0, 255, 0))
        cv2.polylines(bev_map, [ego_corners], True, (255, 255, 255), 2)
        
        # Add grid lines
        self._draw_grid(bev_map)
        
        return bev_map
    
    def _get_box_corners(self, cx, cy, width, length, yaw):
        """Get rotated bounding box corners"""
        # Define corners in local coordinates
        corners_local = np.array([
            [-width/2, -length/2],
            [width/2, -length/2],
            [width/2, length/2],
            [-width/2, length/2]
        ])
        
        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # Rotate and translate
        corners = corners_local @ rot_matrix.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        
        return corners.astype(np.int32)
    
    def _draw_grid(self, bev_map):
        """Draw grid lines on BEV map"""
        # Draw horizontal lines every 10 meters
        for y in range(0, int(self.y_range[1] - self.y_range[0]), 10):
            y_pixel = int(y / self.resolution)
            if y_pixel < self.bev_height:
                cv2.line(bev_map, (0, y_pixel), (self.bev_width, y_pixel), 
                        (50, 50, 50), 1)
        
        # Draw vertical lines every 10 meters
        for x in range(int(self.x_range[0]), int(self.x_range[1]), 10):
            x_pixel = int((x - self.x_range[0]) / self.resolution)
            if 0 <= x_pixel < self.bev_width:
                cv2.line(bev_map, (x_pixel, 0), (x_pixel, self.bev_height), 
                        (50, 50, 50), 1)
