# DotsOCR AMD GPU Deployment Guide

This guide covers deploying DotsOCR on AMD GPUs using ROCm.

## Prerequisites

### Supported AMD GPUs
- **Recommended**: RX 7900 XTX (24GB), RX 7900 XT (20GB)
- **Good**: RX 6900 XT (16GB), RX 6800 XT (16GB)
- **Minimum**: RX 6700 XT (12GB) - may have memory constraints

### Host System Requirements

1. **Install ROCm on host system**:
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/jammy/amdgpu-install_6.4.60403-1_all.deb
sudo dpkg -i amdgpu-install_*.deb
sudo apt update

# Install ROCm
sudo amdgpu-install --usecase=dkms,graphics,opencl,hip,hiplibsdk,rocm

# Add user to groups
sudo usermod -a -G render,video $USER

# Reboot required
sudo reboot
```

2. **Verify ROCm installation**:
```bash
/opt/rocm/bin/rocminfo
/opt/rocm/bin/clinfo
```

## Docker Deployment Options

### Option 1: Full Featured (Recommended)

Uses the complete PyTorch ROCm image with all DotsOCR features:

```bash
# Build and run
docker compose -f docker/docker-compose.amd.yml up --build -d

# Check logs
docker logs dots-ocr-amd-container

# Test health
curl http://localhost:8000/health
```

### Option 2: Simple ROCm Setup

For testing ROCm compatibility:

```bash
# Build with simple ROCm image
docker build -f docker/Dockerfile.rocm-simple -t dots-ocr-amd-simple .

# Run
docker run --rm -p 8000:8000 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  dots-ocr-amd-simple

# Test
curl http://localhost:8000/test
```

## Configuration

### Environment Variables

- `HIP_VISIBLE_DEVICES=0` - Select GPU device
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` - Override GPU architecture if needed
- `ROCM_PATH=/opt/rocm` - ROCm installation path

### Memory Optimization

For GPUs with limited VRAM:

```yaml
# In docker-compose.amd.yml, add:
environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
```

## Usage

### API Endpoints

1. **Health Check**:
```bash
curl http://localhost:8000/health
```

2. **Parse Document**:
```bash
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image>",
    "max_tokens": 1024
  }'
```

### Python Client Example

```python
import base64
import requests
from PIL import Image
import io

# Load and encode image
with open("document.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/parse",
    json={
        "image": image_data,
        "max_tokens": 2048
    }
)

result = response.json()
print(result["result"])
```

## Troubleshooting

### Common Issues

1. **ROCm not detected**:
   - Verify host ROCm installation
   - Check user groups (render, video)
   - Ensure proper device permissions

2. **Out of memory errors**:
   - Reduce max_tokens
   - Use GPU with more VRAM
   - Enable memory optimizations

3. **Performance issues**:
   - Check GPU utilization: `rocm-smi`
   - Monitor memory usage
   - Consider mixed precision training

### Debug Commands

```bash
# Check ROCm devices
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-runtime:6.4.3 rocminfo

# Monitor GPU usage
watch -n 1 rocm-smi

# Container debugging
docker exec -it dots-ocr-amd-container bash
```

## Performance Comparison

| GPU | VRAM | Expected Performance |
|-----|------|---------------------|
| RTX 3070 | 8GB | Baseline (CUDA) |
| RX 6700 XT | 12GB | ~80% of RTX 3070 |
| RX 6800 XT | 16GB | ~90% of RTX 3070 |
| RX 7900 XT | 20GB | ~110% of RTX 3070 |
| RX 7900 XTX | 24GB | ~120% of RTX 3070 |

*Performance varies by workload and ROCm optimization*
