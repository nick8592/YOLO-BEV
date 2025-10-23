# YOLO-BEV

A perception pipeline for Advanced Driver Assistance Systems (ADAS) that combines YOLO-based 2D object detection with 3D bounding box estimation and Bird's Eye View (BEV) transformation using nuScenes front-camera images.

## Documentation

For detailed documentation, see the following files in the `docs/` directory:

- **[PROPOSAL.md](docs/PROPOSAL.md)** - Original project proposal with architecture and methodology
- **[IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** - Implementation details and technical documentation
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Quick reference guide for common tasks
- **[TRAINING_READINESS.md](docs/TRAINING_READINESS.md)** - Training preparation and execution guide
- **[GPU_ACCESS_GUIDE.md](docs/GPU_ACCESS_GUIDE.md)** - GPU setup and troubleshooting solutions
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Complete project structure documentation

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

Verify installation:
```bash
python tests/verify_installation.py
```

### Test Pipeline
Test the full pipeline on a sample:
```bash
python tests/test_pipeline.py
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
├── README.md               # Project overview (you are here)
├── LICENSE                 # MIT License
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation
│
├── configs/               # Configuration files
│   └── config.yaml       # Main pipeline configuration
│
├── docs/                  # Documentation
│   ├── PROPOSAL.md       # Project proposal
│   ├── IMPLEMENTATION.md # Implementation details
│   ├── QUICKSTART.md     # Quick reference
│   └── ...               # Additional guides
│
├── src/                   # Source code
│   ├── pipeline.py       # Main pipeline orchestrator
│   ├── data/             # Dataset loaders
│   ├── models/           # YOLO detector & 3D estimator
│   └── utils/            # BEV transform, visualization, config
│
├── scripts/              # Executable scripts
│   ├── train.py         # Training script
│   ├── inference.py     # Inference script
│   └── demo.py          # Quick demo
│
├── tests/                # Test and verification scripts
│   ├── verify_installation.py
│   ├── test_pipeline.py
│   └── check_training_ready.py
│
├── tools/                # Development tools
│   ├── evaluate.py      # Model evaluation
│   └── visualize_dataset.py
│
├── weights/              # Model weights
│   └── yolov8n.pt       # Pre-trained YOLO weights
│
├── nuscenes/            # nuScenes dataset (symlink)
├── checkpoints/         # Training checkpoints
├── logs/                # Training logs
└── outputs/             # Inference outputs
```

For detailed structure documentation, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).