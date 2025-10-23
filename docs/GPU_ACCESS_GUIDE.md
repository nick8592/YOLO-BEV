# GPU Access Solutions for YOLO-BEV Training

## Current Situation

Your container has CUDA libraries installed (PyTorch 2.0.1+cu117) but cannot access the GPU:
```
✓ PyTorch 2.0.1+cu117 installed
✗ CUDA available: False
```

This means the CUDA toolkit is present, but the container cannot communicate with the GPU hardware.

---

## Solutions (In Order of Recommendation)

### Solution 1: Use NVIDIA Container Toolkit (Recommended)

This is the **best solution** for Docker containers needing GPU access.

#### Check if NVIDIA Docker is installed on host:
```bash
# On your host machine (outside container)
nvidia-smi  # Verify GPU is available on host
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

#### Restart container with GPU access:
```bash
# Stop current container
docker stop <container_name>

# Start with GPU support
docker run --gpus all -it <your_image> /bin/bash

# Or for specific GPU:
docker run --gpus '"device=0"' -it <your_image> /bin/bash
```

#### If NVIDIA Docker not installed on host:
```bash
# Install NVIDIA Container Toolkit (on host)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

### Solution 2: Use VS Code Dev Container with GPU Support

If using VS Code with Dev Containers:

#### Edit `.devcontainer/devcontainer.json`:
```json
{
  "name": "YOLO-BEV",
  "image": "your-image",
  "runArgs": [
    "--gpus=all"
  ],
  "hostRequirements": {
    "gpu": true
  }
}
```

#### Rebuild container:
```
Ctrl+Shift+P -> "Dev Containers: Rebuild Container"
```

---

### Solution 3: Cloud GPU Options

If your local machine doesn't have GPU or has setup issues:

#### Option A: Google Colab (Free/Pro)
- **Pros:** Free GPU access, Jupyter notebooks
- **Cons:** Session timeouts, limited storage
- **Setup:**
  ```python
  # In Colab notebook
  !git clone https://github.com/nick8592/YOLO-BEV.git
  %cd YOLO-BEV
  !pip install -r requirements.txt
  # Upload nuScenes data or mount Google Drive
  ```

#### Option B: AWS EC2 GPU Instances
- **Recommended:** g4dn.xlarge ($0.526/hour)
- **GPU:** NVIDIA T4 (16GB)
- **Setup:** Launch with Deep Learning AMI
- **Cost:** ~$5-10 for full training run

#### Option C: Paperspace Gradient
- **Pros:** Easy setup, Jupyter notebooks
- **Pricing:** $0.51/hour for P5000
- **Good for:** Medium-scale training

#### Option D: Lambda Labs
- **Pros:** Best price/performance
- **GPU:** A100, A6000 available
- **Cost:** $0.50-1.10/hour

---

### Solution 4: Train on CPU (Current Setup)

If GPU access isn't immediately available, you can **test the training pipeline on CPU**:

#### Advantages:
✅ Verify training code works
✅ Debug any issues
✅ Generate initial checkpoints
✅ Validate data pipeline

#### Already Configured:
```yaml
# configs/config.yaml
device: "cpu"
batch_size: 4      # Reduced for CPU
num_epochs: 10     # Test run
```

#### Start test training:
```bash
python3 scripts/train.py --config configs/config.yaml --epochs 5
```

#### Then later:
- Resume training on GPU from checkpoint
- Transfer checkpoint to GPU machine
- Continue training with full epochs

---

## Verification Commands

### Check GPU access in container:
```bash
# Inside container
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
nvidia-smi  # Should show GPU if accessible
```

### Check from host:
```bash
# Outside container
nvidia-smi  # Verify GPU works on host
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

---

## Recommended Workflow

### Immediate Actions (Today):

1. **Test training pipeline on CPU:**
   ```bash
   python3 scripts/train.py --config configs/config.yaml --epochs 2
   ```
   - Verifies code works
   - Takes ~20-30 minutes for 2 epochs
   - Generates initial checkpoint

2. **Verify training runs without errors:**
   - Check loss decreases
   - Checkpoint saves correctly
   - No code bugs

### Next Steps (When GPU Available):

3. **Setup GPU access:**
   - Restart container with `--gpus all`
   - Or use cloud GPU instance
   - Verify with `nvidia-smi`

4. **Update configuration for GPU:**
   ```yaml
   device: "cuda"
   batch_size: 16    # Increase for GPU
   num_epochs: 100   # Full training
   ```

5. **Resume training from checkpoint:**
   ```bash
   python3 scripts/train.py --config configs/config.yaml \
       --resume checkpoints/checkpoint_epoch_2.pth
   ```

---

## Training Time Comparison

| Setup | Device | Batch Size | Time per Epoch | 100 Epochs |
|-------|--------|------------|----------------|------------|
| Current | CPU | 4 | ~50 min | ~83 hours ⚠️ |
| Local GPU | RTX 3090 | 16 | ~3 min | ~5 hours ✅ |
| Cloud GPU | T4 | 16 | ~5 min | ~8 hours ✅ |
| Cloud GPU | A100 | 32 | ~1.5 min | ~2.5 hours ✅ |

---

## Quick GPU Setup Scripts

### Check GPU availability:
```bash
#!/bin/bash
# check_gpu.sh

echo "=== Host GPU Check ==="
nvidia-smi

echo -e "\n=== Docker GPU Check ==="
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi

echo -e "\n=== PyTorch GPU Check ==="
docker run --rm --gpus all python:3.8 bash -c "
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117 -q
python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'
"
```

### Restart container with GPU:
```bash
#!/bin/bash
# restart_with_gpu.sh

CONTAINER_ID=$(docker ps -q --filter "name=yolo-bev")
IMAGE_NAME=$(docker inspect --format='{{.Config.Image}}' $CONTAINER_ID)

echo "Stopping container..."
docker stop $CONTAINER_ID

echo "Starting with GPU access..."
docker run --gpus all -it --name yolo-bev-gpu \
    -v $(pwd):/workspace \
    $IMAGE_NAME
```

---

## Troubleshooting

### Error: "could not select device driver"
**Solution:** Install NVIDIA Docker on host machine

### Error: "NVIDIA-SMI has failed"
**Solution:** Update NVIDIA drivers on host

### Error: "no CUDA-capable device is detected"
**Causes:**
1. Container not started with `--gpus` flag
2. NVIDIA Docker not installed
3. GPU drivers not installed on host

### Container has GPU but PyTorch doesn't see it:
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

---

## My Recommendation for You

### Short Term (Now):
1. ✅ **Run 2-5 epochs on CPU** to verify pipeline works
2. ✅ Check that training completes without errors
3. ✅ Validate checkpoint saving/loading

### Medium Term (This Week):
1. **Setup GPU access** using one of:
   - Restart container with `--gpus all` (if GPU on host)
   - Use Google Colab for free testing
   - Rent cloud GPU for $5-10

### Long Term (Production):
1. **Full training** with 100 epochs on GPU
2. Evaluate model on validation set
3. Deploy trained model for inference

---

## Cost Analysis

### Free Options:
- **CPU training:** Free but very slow (83 hours)
- **Google Colab:** Free GPU for 12 hours/day
  - Limited: May disconnect
  - Good for: Testing, small runs

### Paid Options:
- **AWS g4dn.xlarge:** ~$5-10 for full training
- **Paperspace:** ~$4-8 for full training
- **Lambda Labs:** ~$2-5 for full training (best value)

### My Recommendation:
**Lambda Labs or Paperspace for $5-10 one-time cost** is worth it vs 83 hours on CPU.

---

## Summary

**Current Status:** ✅ Ready to train on CPU (test run)

**Best Options:**
1. 🏆 **Restart container with GPU** (if available on host)
2. 🥈 **Rent cloud GPU for $5-10** (Lambda Labs/Paperspace)
3. 🥉 **Google Colab** (free but limited)
4. ⚠️ **CPU training** (works but very slow)

**Immediate Action:**
```bash
# Test on CPU now (20-30 min for 2 epochs)
python3 scripts/train.py --config configs/config.yaml --epochs 2

# Then setup GPU access and resume training
```

The pipeline is ready - GPU is just an optimization for speed! 🚀
