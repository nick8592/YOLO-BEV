# YOLO-BEV

A perception pipeline for Advanced Driver Assistance Systems (ADAS) that combines YOLO-based 2D object detection with 3D bounding box estimation and Bird's Eye View (BEV) transformation using nuScenes front-camera images.

## Project Proposal

For detailed information about the pipeline architecture, methodology, and implementation plan, see [PROPOSAL.md](PROPOSAL.md).

For implementation details, usage examples, and technical documentation, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Quick Start

### Data Setup
Create a symbolic link to the nuScenes dataset:
```bash
ln -s /home/nick/Documents/dataset/nuscenes nuscenes
```

### Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Run Demo
Quick demo on sample nuScenes data:
```bash
python scripts/demo.py
```

### Inference
Run inference on a single image:
```bash
python scripts/inference.py --image path/to/image.jpg --output outputs/inference
```

### Training
Train the 3D estimation module:
```bash
python scripts/train.py --config configs/config.yaml --epochs 100
```

## Features

- Monocular 3D object detection from front-camera images
- YOLO-powered 2D detection with 3D regression
- BEV transformation for spatial planning
- Real-time performance optimized for ADAS applications

## Project Structure

```
YOLO-BEV/
├── configs/            # Configuration files
│   └── config.yaml
├── src/                # Source code
│   ├── data/          # Dataset loaders
│   ├── models/        # Model definitions
│   ├── utils/         # Utility functions
│   └── pipeline.py    # Main pipeline
├── scripts/           # Training and inference scripts
│   ├── demo.py       # Quick demo script
│   ├── inference.py  # Inference script
│   └── train.py      # Training script
├── nuscenes/         # nuScenes dataset (symlink)
├── outputs/          # Output visualizations
├── checkpoints/      # Model checkpoints
└── requirements.txt  # Python dependencies
```