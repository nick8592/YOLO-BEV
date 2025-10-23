#!/usr/bin/env python
"""
Inference script for YOLO-BEV Pipeline
Run inference on a single image or directory of images with ground truth comparison
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import YOLOBEVPipeline
from src.utils.config import load_config
from nuscenes.nuscenes import NuScenes


def main():
    parser = argparse.ArgumentParser(description='YOLO-BEV Inference')
    parser.add_argument('--image', type=str, required=False,
                       help='Path to input image or directory')
    parser.add_argument('--nuscenes', action='store_true',
                       help='Use nuScenes dataset for inference with GT comparison')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of nuScenes samples to process (default: 5)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--output', type=str, default='outputs/inference',
                       help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Display results in window')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize pipeline
    print("Initializing YOLO-BEV pipeline...")
    pipeline = YOLOBEVPipeline(config_path=args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        pipeline.load_checkpoint(args.checkpoint)
    
    # Process based on mode
    if args.nuscenes:
        # Process nuScenes samples with GT comparison
        process_nuscenes(pipeline, args.config, args.num_samples, args.output, args.show)
    elif args.image:
        # Process image(s)
        if os.path.isfile(args.image):
            # Single image
            process_single_image(pipeline, args.image, args.output, args.show)
        elif os.path.isdir(args.image):
            # Directory of images
            process_directory(pipeline, args.image, args.output, args.show)
        else:
            print(f"Error: {args.image} is not a valid file or directory")
            return
    else:
        print("Error: Must specify either --image or --nuscenes")
        return
    
    print("Inference complete!")


def process_single_image(pipeline, image_path, output_dir, show=False):
    """Process a single image"""
    print(f"\nProcessing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run pipeline
    results = pipeline.process_image(image)
    
    # Visualize
    visualization = pipeline.visualize_results(image, results)
    
    # Save
    output_path = os.path.join(
        output_dir,
        f"{Path(image_path).stem}_result.jpg"
    )
    cv2.imwrite(output_path, visualization)
    print(f"Saved visualization to: {output_path}")
    
    # Display if requested
    if show:
        cv2.imshow('YOLO-BEV Results', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print statistics
    print(f"  - 2D detections: {len(results['detections_2d']['boxes'])}")
    print(f"  - 3D boxes: {len(results['boxes_3d'])}")
    print(f"  - BEV boxes: {len(results['boxes_bev'])}")


def process_directory(pipeline, image_dir, output_dir, show=False):
    """Process all images in a directory"""
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for image_path in image_files:
        process_single_image(pipeline, str(image_path), output_dir, show)


def process_nuscenes(pipeline, config_path, num_samples, output_dir, show=False):
    """Process nuScenes samples with ground truth comparison"""
    print(f"\nProcessing {num_samples} nuScenes samples with GT comparison...")
    
    # Load config
    config = load_config(config_path)
    
    # Initialize nuScenes
    print("Loading nuScenes dataset...")
    nusc = NuScenes(
        version=config['dataset']['version'],
        dataroot=config['dataset']['data_root'],
        verbose=False
    )
    
    # Get random samples
    import random
    sample_indices = random.sample(range(len(nusc.sample)), min(num_samples, len(nusc.sample)))
    
    print(f"\nVisualizing {len(sample_indices)} samples...\n")
    
    for idx, sample_idx in enumerate(sample_indices, 1):
        sample = nusc.sample[sample_idx]
        sample_token = sample['token']
        
        print(f"Processing sample {idx}/{len(sample_indices)}: {sample_token}")
        
        # Get front camera data
        cam_token = sample['data']['CAM_FRONT']
        sample_data = nusc.get('sample_data', cam_token)
        
        # Get ground truth
        bev_gt, boxes_3d_gt, boxes_2d_gt, class_ids_gt, image_path, camera_intrinsic = \
            get_ground_truth_bev(nusc, sample_token, config)
        
        # Load image
        image = cv2.imread(image_path)
        
        # Create GT combined visualization (image with 3D boxes + GT BEV)
        gt_combined = create_gt_combined_visualization(
            image, bev_gt, boxes_2d_gt, class_ids_gt
        )
        
        # Run inference
        print("  Running inference...")
        results = pipeline.process_image(image)
        
        # Create prediction combined visualization
        pred_combined = create_pred_combined_visualization(
            image, results, camera_intrinsic, config
        )
        
        # Create comparison (predicted vs GT BEV)
        comparison = create_comparison_visualization(
            results['bev_map'], bev_gt, config
        )
        
        # Save outputs
        base_name = f"sample_{idx}"
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_gt_combined.jpg"), gt_combined)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_pred_combined.jpg"), pred_combined)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_comparison.jpg"), comparison)
        
        print(f"  Ground truth boxes: {len(boxes_3d_gt)}")
        print(f"  Predicted boxes: {len(results['boxes_3d'])}")
        print(f"  Saved: {base_name}_*.jpg\n")
        
        if show:
            cv2.imshow('GT Combined', gt_combined)
            cv2.imshow('Prediction Combined', pred_combined)
            cv2.imshow('Comparison', comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print(f"\nVisualization complete! Results saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - sample_X_gt_combined.jpg (camera with GT 3D boxes + GT BEV)")
    print("  - sample_X_pred_combined.jpg (camera with predicted 3D boxes + predicted BEV)")
    print("  - sample_X_comparison.jpg (predicted vs ground truth BEV)")


def get_ground_truth_bev(nusc, sample_token, config):
    """Extract ground truth BEV from nuScenes"""
    from src.utils.bev_transform import BEVTransform
    
    # Get sample and camera data
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    sample_data = nusc.get('sample_data', cam_token)
    
    # Get camera calibration
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    camera_intrinsic = np.array(cs_record['camera_intrinsic'])
    cam_translation = np.array(cs_record['translation'])
    cam_rotation = Quaternion(cs_record['rotation'])
    ego_translation = np.array(pose_record['translation'])
    ego_rotation = Quaternion(pose_record['rotation'])
    
    # Get 3D boxes
    boxes = nusc.get_boxes(sample_data['token'])
    print(f"  Total boxes from nuScenes: {len(boxes)}")
    
    # Class mapping (nuScenes -> COCO-like)
    class_mapping = {
        'human.pedestrian': 0,  # person
        'vehicle.car': 2,       # car
        'vehicle.motorcycle': 3,# motorcycle
        'vehicle.bus': 5,       # bus
        'vehicle.truck': 7,     # truck
    }
    
    boxes_3d_gt = []
    boxes_2d_gt = []
    class_ids_gt = []
    
    for box in boxes:
        # Get main class
        main_class = '.'.join(box.name.split('.')[:2])
        
        # Map to COCO class if possible
        class_id = class_mapping.get(main_class, None)
        if class_id is None:
            continue
        
        # Check visibility
        visibility = nusc.get('sample_annotation', box.token)['visibility_token']
        visibility_desc = nusc.get('visibility', visibility)['description']
        if visibility_desc == '0-40':
            continue
        
        # Get box corners in global coordinates (3x8 array)
        corners_global = box.corners()
        
        # Transform corners from global to ego to camera coordinates
        corners_ego = corners_global - ego_translation.reshape(3, 1)
        corners_ego = ego_rotation.inverse.rotation_matrix @ corners_ego
        corners_cam = corners_ego - cam_translation.reshape(3, 1)
        corners_cam = cam_rotation.inverse.rotation_matrix @ corners_cam
        
        # Get box center in camera frame
        box_center_cam = corners_cam.mean(axis=1)
        x, y, z = box_center_cam
        
        # Skip boxes behind camera or very far
        if z < 0 or z > 100:
            continue
        
        # Project corners to 2D
        corners_2d = project_3d_to_2d(corners_cam.T, camera_intrinsic)
        
        if corners_2d is not None:
            # Get dimensions from corners
            w = corners_cam[0].max() - corners_cam[0].min()
            h = corners_cam[1].max() - corners_cam[1].min()
            l = corners_cam[2].max() - corners_cam[2].min()
            
            boxes_3d_gt.append([x, y, z, w, h, l, 0])
            boxes_2d_gt.append(corners_2d)
            class_ids_gt.append(class_id)
    
    print(f"  Extracted {len(boxes_3d_gt)} boxes with matching classes")
    
    # Create BEV map from ground truth
    bev_transform = BEVTransform(config)
    
    # Convert 3D boxes to BEV coordinates
    boxes_bev_gt = bev_transform.camera_to_bev(boxes_3d_gt)
    
    # Create BEV visualization
    bev_gt = bev_transform.create_bev_map(boxes_bev_gt, class_ids_gt)
    
    # Get image path
    image_path = os.path.join(nusc.dataroot, sample_data['filename'])
    
    return bev_gt, boxes_3d_gt, boxes_2d_gt, class_ids_gt, image_path, camera_intrinsic


def project_3d_to_2d(points_3d, camera_intrinsic, image_width=1600, image_height=900):
    """Project 3D points to 2D image plane and check if visible"""
    if np.any(points_3d[:, 2] < 0.1):
        return None
    
    # Project to 2D
    points_2d = points_3d @ camera_intrinsic.T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    
    # Check if at least some corners are within the image frame
    margin = 50
    x_coords = points_2d[:, 0]
    y_coords = points_2d[:, 1]
    
    visible = np.any((x_coords > -margin) & (x_coords < image_width + margin) & 
                     (y_coords > -margin) & (y_coords < image_height + margin))
    
    if not visible:
        return None
    
    return points_2d


def draw_3d_box_on_image(image, corners_2d, color, thickness=2):
    """Draw 3D bounding box edges on image"""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    for start, end in edges:
        pt1 = tuple(corners_2d[start].astype(int))
        pt2 = tuple(corners_2d[end].astype(int))
        cv2.line(image, pt1, pt2, color, thickness)


def create_gt_combined_visualization(image, bev_gt, boxes_2d_gt, class_ids_gt):
    """Create combined visualization with GT 3D boxes on image and GT BEV"""
    # Class colors (BGR)
    class_colors = {
        0: (0, 0, 255),    # person - red
        2: (255, 0, 0),    # car - blue
        3: (255, 0, 255),  # motorcycle - magenta
        5: (255, 255, 0),  # bus - cyan
        7: (255, 128, 0),  # truck - orange
    }
    
    # Draw 3D boxes on image
    image_with_boxes = image.copy()
    for corners_2d, class_id in zip(boxes_2d_gt, class_ids_gt):
        color = class_colors.get(class_id, (0, 255, 0))
        draw_3d_box_on_image(image_with_boxes, corners_2d, color, thickness=2)
    
    # BEV is already in BGR format from create_bev_map
    bev_vis = bev_gt
    bev_height = image.shape[0]
    bev_width = int(bev_vis.shape[1] * bev_height / bev_vis.shape[0])
    bev_resized = cv2.resize(bev_vis, (bev_width, bev_height))
    
    # Add title
    cv2.putText(image_with_boxes, "Ground Truth 3D Boxes", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(bev_resized, "Ground Truth BEV", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Concatenate horizontally
    combined = np.hstack([image_with_boxes, bev_resized])
    return combined


def create_pred_combined_visualization(image, results, camera_intrinsic, config):
    """Create combined visualization with predicted 3D boxes on image and predicted BEV"""
    from src.utils.visualization import Visualizer
    
    # Draw 3D boxes and BEV
    visualizer = Visualizer(config)
    image_with_boxes = visualizer.draw_3d_boxes(
        image.copy(), 
        results['boxes_3d'], 
        camera_intrinsic
    )
    
    # Get BEV map (already in BGR format)
    bev_vis = results['bev_map']
    bev_height = image.shape[0]
    bev_width = int(bev_vis.shape[1] * bev_height / bev_vis.shape[0])
    bev_resized = cv2.resize(bev_vis, (bev_width, bev_height))
    
    # Add title
    cv2.putText(image_with_boxes, "Predicted 3D Boxes", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(bev_resized, "Predicted BEV", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Concatenate horizontally
    combined = np.hstack([image_with_boxes, bev_resized])
    return combined


def create_comparison_visualization(bev_pred, bev_gt, config):
    """Create side-by-side comparison of predicted and GT BEV"""
    from src.utils.visualization import Visualizer
    
    visualizer = Visualizer(config)
    comparison = visualizer.create_bev_comparison(bev_pred, bev_gt)
    return comparison


if __name__ == '__main__':
    main()
