# YOLO-BEV Quick Reference

## Installation
```bash
# Clone repository
git clone https://github.com/nick8592/YOLO-BEV.git
cd YOLO-BEV

# Link nuScenes dataset
ln -s /path/to/nuscenes nuscenes

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Demo
```bash
python scripts/demo.py
```

### Inference on Single Image
```bash
python scripts/inference.py --image test.jpg --output outputs/
```

### Inference on Directory
```bash
python scripts/inference.py --image /path/to/images/ --output outputs/
```

### Training
```bash
python scripts/train.py --config configs/config.yaml
```

### Resume Training
```bash
python scripts/train.py --resume checkpoints/checkpoint_epoch_50.pth
```

## Python API

### Basic Pipeline
```python
from src.pipeline import YOLOBEVPipeline
import cv2

# Initialize
pipeline = YOLOBEVPipeline('configs/config.yaml')

# Process image
image = cv2.imread('test.jpg')
results = pipeline.process_image(image)

# Visualize
vis = pipeline.visualize_results(image, results)
cv2.imwrite('result.jpg', vis)
```

### Results Structure
```python
results = {
    'detections_2d': {
        'boxes': np.array,      # Shape: [N, 4]
        'scores': np.array,     # Shape: [N]
        'classes': np.array     # Shape: [N]
    },
    'boxes_3d': list,          # [x, y, z, w, h, l, yaw]
    'boxes_bev': list,         # [x_bev, y_bev, w, l, yaw]
    'bev_map': np.array        # Shape: [H, W, 3]
}
```

### Load Checkpoint
```python
pipeline.load_checkpoint('checkpoints/model.pth')
```

### Save Checkpoint
```python
pipeline.save_checkpoint('checkpoints/model.pth', epoch=10)
```

## Configuration

### Key Config Parameters
```yaml
# configs/config.yaml

dataset:
  data_root: "./nuscenes"
  version: "v1.0-mini"

yolo:
  model_name: "yolov8n"      # yolov8n/s/m/l/x
  confidence_threshold: 0.25
  iou_threshold: 0.45

estimation_3d:
  backbone: "resnet50"        # resnet50/101, vgg16/19, efficientnet_b0
  
bev:
  x_range: [-50, 50]         # meters
  y_range: [0, 100]
  resolution: 0.2            # meters/pixel

training:
  batch_size: 8
  learning_rate: 0.001
```

## Common Commands

### Check GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Count Parameters
```python
from src.models.estimator_3d import Estimator3D
model = Estimator3D(config)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")
```

### Visualize BEV Only
```python
from src.utils.bev_transform import BEVTransform
from src.utils.visualization import Visualizer

bev = BEVTransform(config)
vis = Visualizer(config)

boxes_bev = bev.camera_to_bev(boxes_3d)
bev_map = bev.create_bev_map(boxes_bev, class_ids)
vis.visualize_bev(bev_map)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use smaller YOLO model (yolov8n)
- Use smaller backbone (resnet50 instead of resnet101)

### No Objects Detected
- Lower confidence threshold in config
- Check if nuScenes classes match YOLO classes
- Verify image is loaded correctly

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### Dataset Not Found
```bash
ls -la nuscenes  # Should show symlink
# If not, recreate:
ln -s /path/to/nuscenes nuscenes
```

## File Locations

- **Config**: `configs/config.yaml`
- **Checkpoints**: `checkpoints/`
- **Outputs**: `outputs/`
- **Logs**: `logs/`
- **Models**: `src/models/`
- **Utils**: `src/utils/`

## Dependencies

Core:
- PyTorch >= 2.0.0
- Ultralytics >= 8.0.0 (YOLO)
- nuscenes-devkit >= 1.1.9
- OpenCV >= 4.8.0

Full list: `requirements.txt`

## Performance Tips

1. Use GPU for inference
2. Batch process multiple images
3. Cache YOLO model on first load
4. Use smaller image resolution for faster processing
5. Use FP16 precision for inference

## Support

- Documentation: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- Proposal: [PROPOSAL.md](PROPOSAL.md)
- Issues: GitHub Issues
- Repository: https://github.com/nick8592/YOLO-BEV
