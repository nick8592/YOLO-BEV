# YOLO-BEV Project Structure

This document describes the organization and purpose of each directory and key file in the YOLO-BEV project.

## Directory Structure

```
YOLO-BEV/
├── README.md                   # Project overview and quick start
├── LICENSE                     # Project license (MIT)
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation configuration
│
├── configs/                    # Configuration files
│   └── config.yaml            # Main pipeline configuration
│
├── docs/                       # Documentation
│   ├── PROPOSAL.md            # Original project proposal
│   ├── IMPLEMENTATION.md      # Implementation details
│   ├── QUICKSTART.md          # Quick reference guide
│   ├── TRAINING_READINESS.md  # Training preparation guide
│   ├── GPU_ACCESS_GUIDE.md    # GPU setup and troubleshooting
│   ├── VERIFICATION_REPORT.md # Installation verification results
│   └── images/                # Documentation images and diagrams
│
├── src/                        # Source code (main package)
│   ├── __init__.py
│   ├── pipeline.py            # Main YOLOBEVPipeline orchestrator
│   │
│   ├── models/                # Neural network models
│   │   ├── __init__.py
│   │   ├── yolo_detector.py   # YOLO 2D object detection wrapper
│   │   └── estimator_3d.py    # 3D bounding box estimation network
│   │
│   ├── data/                  # Dataset loaders and data processing
│   │   ├── __init__.py
│   │   └── nuscenes_dataset.py # nuScenes dataset PyTorch loader
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── config.py          # Configuration loading
│       ├── bev_transform.py   # Camera to BEV coordinate transformation
│       └── visualization.py   # Visualization utilities
│
├── scripts/                    # Executable scripts
│   ├── train.py               # Training script for 3D estimator
│   ├── inference.py           # Run inference on images/videos
│   └── demo.py                # Quick demo script
│
├── tests/                      # Test scripts
│   ├── __init__.py
│   ├── verify_installation.py # Check all dependencies installed
│   ├── test_pipeline.py       # Test full pipeline on sample image
│   └── check_training_ready.py # Validate training setup
│
├── tools/                      # Development and analysis tools
│   ├── __init__.py
│   ├── evaluate.py            # Model evaluation on test set
│   └── visualize_dataset.py   # Visualize dataset samples
│
├── weights/                    # Pre-trained model weights
│   ├── README.md              # Weights documentation
│   ├── yolov8n.pt            # YOLOv8 nano pre-trained weights (6.2MB)
│   └── .gitkeep              # Preserve directory in git
│
├── data/                       # Dataset storage (gitignored)
│   └── .gitkeep              # Preserve directory structure
│
├── nuscenes/                   # nuScenes dataset (symlink)
│   └── [symlink to dataset]
│
├── checkpoints/                # Training checkpoints (gitignored)
│   └── .gitkeep
│
├── logs/                       # Training logs (gitignored)
│   └── .gitkeep
│
└── outputs/                    # Inference outputs (gitignored)
    └── .gitkeep
```

---

## Directory Descriptions

### 📁 `src/` - Source Code
The main Python package containing all pipeline components.

**Key Files:**
- `pipeline.py`: Main orchestrator that coordinates YOLO detection, 3D estimation, and BEV transformation
- `models/yolo_detector.py`: Wrapper for Ultralytics YOLOv8 with custom inference methods
- `models/estimator_3d.py`: Custom 3D bounding box estimation network (ResNet/VGG/EfficientNet backbones)
- `data/nuscenes_dataset.py`: PyTorch Dataset class for loading nuScenes data with 3D annotations
- `utils/bev_transform.py`: Transforms 3D boxes from camera coordinates to bird's eye view
- `utils/visualization.py`: Visualization utilities for 2D/3D boxes and BEV maps
- `utils/config.py`: Configuration file loading and validation

**Usage:**
```python
from src.pipeline import YOLOBEVPipeline
from src.utils.config import load_config

config = load_config('configs/config.yaml')
pipeline = YOLOBEVPipeline(config)
```

---

### 📁 `scripts/` - Executable Scripts
High-level scripts for training, inference, and demos.

**Key Files:**
- `train.py`: Train the 3D estimator on nuScenes dataset
- `inference.py`: Run inference on custom images or videos
- `demo.py`: Quick demonstration of the pipeline

**Usage:**
```bash
# Training
python scripts/train.py --config configs/config.yaml --epochs 100

# Inference
python scripts/inference.py --image path/to/image.jpg --output outputs/result.jpg

# Demo
python scripts/demo.py
```

---

### 📁 `tests/` - Test Suite
Verification and testing scripts to ensure system is working correctly.

**Key Files:**
- `verify_installation.py`: Check all dependencies are installed
- `test_pipeline.py`: Test full pipeline on sample nuScenes image
- `check_training_ready.py`: Comprehensive check before starting training

**Usage:**
```bash
# Verify installation
python tests/verify_installation.py

# Test pipeline
python tests/test_pipeline.py

# Check training readiness
python tests/check_training_ready.py
```

---

### 📁 `tools/` - Development Tools
Additional utilities for evaluation, visualization, and analysis.

**Key Files:**
- `evaluate.py`: Evaluate trained model on validation/test set with metrics
- `visualize_dataset.py`: Visualize random samples from nuScenes dataset

**Usage:**
```bash
# Evaluate model
python tools/evaluate.py --checkpoint checkpoints/best_model.pth --split val

# Visualize dataset
python tools/visualize_dataset.py --samples 20 --output outputs/dataset_viz/
```

---

### 📁 `configs/` - Configuration Files
YAML configuration files for all pipeline parameters.

**Key File:**
- `config.yaml`: Main configuration with:
  - Model architecture settings (YOLO model, 3D estimator backbone)
  - Training hyperparameters (learning rate, batch size, epochs)
  - Dataset paths and versions
  - BEV transformation parameters
  - Inference settings

**Usage:**
```yaml
# configs/config.yaml
model:
  yolo_model: 'yolov8n'
  backbone: 'resnet50'

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
  device: 'cuda'
```

---

### 📁 `docs/` - Documentation
Comprehensive project documentation and guides.

**Key Files:**
- `PROPOSAL.md`: Original project proposal with methodology
- `IMPLEMENTATION.md`: Detailed implementation documentation
- `QUICKSTART.md`: Quick reference for common tasks
- `TRAINING_READINESS.md`: Guide for preparing and starting training
- `GPU_ACCESS_GUIDE.md`: Solutions for GPU access issues
- `VERIFICATION_REPORT.md`: Results from installation verification

---

### 📁 `weights/` - Model Weights
Pre-trained model weights and trained checkpoints.

**Contents:**
- `yolov8n.pt`: Pre-trained YOLOv8 nano model (6.2MB)
- Future: Custom trained 3D estimator weights

**Note:** Large checkpoint files (>10MB) are gitignored by default.

---

### 📁 `data/` - Additional Datasets
Storage for additional datasets beyond nuScenes (gitignored).

**Note:** The main nuScenes dataset is symlinked at `./nuscenes/`

---

### 📁 `checkpoints/` - Training Checkpoints
Training checkpoints saved during model training (gitignored).

**Generated Files:**
- `checkpoint_epoch_{N}.pth`: Checkpoint after epoch N
- `best_model.pth`: Best model based on validation loss
- `latest.pth`: Most recent checkpoint

---

### 📁 `logs/` - Training Logs
Training logs and metrics (gitignored).

**Generated Files:**
- `training.log`: Text log of training progress
- TensorBoard logs (if enabled)

---

### 📁 `outputs/` - Inference Outputs
Inference results, visualizations, and test outputs (gitignored).

**Generated Files:**
- Visualization images
- BEV maps
- Detection results

---

## Key Files at Root

### `README.md`
Main project documentation with:
- Project overview
- Features and capabilities
- Installation instructions
- Quick start guide
- Repository structure

### `requirements.txt`
Python package dependencies:
```
torch>=2.0.0
torchvision
ultralytics>=8.0.0
nuscenes-devkit
opencv-python
numpy
matplotlib
pyyaml
tqdm
scipy
```

### `setup.py`
Package installation configuration for `pip install -e .`

### `LICENSE`
MIT License for the project

### `.gitignore`
Git ignore patterns for:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- Large model files (except small YOLO models)
- Generated outputs (logs, checkpoints, visualizations)
- Dataset files (except configuration)

---

## Import Conventions

### Importing from `src/`
```python
# From scripts/ or tests/
from src.pipeline import YOLOBEVPipeline
from src.models.yolo_detector import YOLODetector
from src.models.estimator_3d import Estimator3D
from src.data.nuscenes_dataset import NuScenesDataset
from src.utils.config import load_config
from src.utils.visualization import Visualizer
```

### Running Scripts
```bash
# From project root
python scripts/train.py
python scripts/inference.py
python tests/verify_installation.py
python tools/evaluate.py
```

---

## Development Workflow

### 1. Initial Setup
```bash
pip install -r requirements.txt
python tests/verify_installation.py
```

### 2. Test Pipeline
```bash
python tests/test_pipeline.py
```

### 3. Training
```bash
python tests/check_training_ready.py
python scripts/train.py --config configs/config.yaml
```

### 4. Evaluation
```bash
python tools/evaluate.py --checkpoint checkpoints/best_model.pth --split val
```

### 5. Inference
```bash
python scripts/inference.py --image path/to/image.jpg
```

---

## Adding New Components

### Adding a New Model
1. Create file in `src/models/new_model.py`
2. Implement model class
3. Update `src/models/__init__.py`
4. Add to pipeline in `src/pipeline.py`

### Adding a New Tool
1. Create file in `tools/new_tool.py`
2. Add argparse for CLI
3. Import from `src/` modules
4. Document in this file

### Adding Tests
1. Create test file in `tests/test_*.py`
2. Import from `src/` modules
3. Run with `python tests/test_*.py`

---

## File Naming Conventions

- **Python modules**: `lowercase_with_underscores.py`
- **Classes**: `CamelCase`
- **Functions**: `lowercase_with_underscores()`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Config files**: `lowercase.yaml`
- **Documentation**: `UPPERCASE.md`

---

## Git Workflow

### What's Tracked
✅ Source code (`src/`, `scripts/`, `tests/`, `tools/`)
✅ Configuration files (`configs/`)
✅ Documentation (`docs/`, `README.md`)
✅ Small model files (`weights/yolov8n.pt` ~6MB)
✅ Requirements and setup files

### What's Ignored
❌ Generated outputs (`outputs/`, `logs/`, `checkpoints/`)
❌ Large model files (`*.pth` checkpoints)
❌ Dataset files (`nuscenes/`, `data/`)
❌ Python cache (`__pycache__/`, `*.pyc`)
❌ Virtual environments

---

## Best Practices

1. **Always run from project root**: `python scripts/train.py` not `cd scripts && python train.py`
2. **Use configuration files**: Modify `configs/config.yaml` instead of hardcoding parameters
3. **Test before training**: Run `tests/check_training_ready.py` to validate setup
4. **Use relative imports**: Import from `src.` rather than absolute paths
5. **Document changes**: Update relevant `.md` files when adding features
6. **Git ignore large files**: Never commit large checkpoints or datasets

---

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Verify installation | `python tests/verify_installation.py` |
| Test pipeline | `python tests/test_pipeline.py` |
| Check training ready | `python tests/check_training_ready.py` |
| Train model | `python scripts/train.py --config configs/config.yaml` |
| Run inference | `python scripts/inference.py --image IMAGE_PATH` |
| Evaluate model | `python tools/evaluate.py --checkpoint CHECKPOINT_PATH` |
| Visualize dataset | `python tools/visualize_dataset.py --samples 10` |

---

## Summary

The YOLO-BEV project is organized for:
- **Clear separation of concerns**: Source code, scripts, tests, tools, docs
- **Easy development**: Well-defined imports and directory structure
- **Git-friendly**: Proper ignoring of generated files and large datasets
- **Scalability**: Easy to add new models, tools, and features
- **Maintainability**: Comprehensive documentation and consistent naming

For more information, see the documentation in the `docs/` directory.
