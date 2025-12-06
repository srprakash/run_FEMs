# TensorRT/ONNX Optimization Guide

## Overview

TensorRT and ONNX Runtime are optimization frameworks that can significantly speed up YOLO inference by:
- **Graph optimization**: Fusing operations, eliminating redundant computations
- **Hardware-specific optimizations**: Leveraging GPU tensor cores, optimized kernels
- **Quantization**: Using INT8/FP16 precision for faster inference
- **Operator fusion**: Combining multiple operations into single kernels

## Performance Comparison

| Backend | Speed (FPS) | Speedup | Notes |
|---------|-------------|---------|-------|
| PyTorch (CPU) | ~11 FPS | 1x | Baseline |
| PyTorch (GPU) | ~30 FPS | 2.7x | With CUDA |
| ONNX Runtime (CPU) | ~15 FPS | 1.4x | Optimized CPU kernels |
| ONNX Runtime (GPU) | ~45 FPS | 4x | CUDA execution provider |
| TensorRT | ~60-100 FPS | 5-9x | NVIDIA GPUs only, requires conversion |

## ONNX Runtime Implementation

### Step 1: Install ONNX Runtime

```bash
# For CPU only
pip install onnxruntime

# For GPU support (CUDA)
pip install onnxruntime-gpu

# For TensorRT support (NVIDIA GPUs)
pip install onnxruntime-gpu tensorrt
```

### Step 2: Convert YOLO Model to ONNX

```python
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Export to ONNX
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,      # Simplify graph for better performance
    opset=12,           # ONNX opset version
    dynamic=False,      # Fixed input size (better optimization)
    half=False          # FP32 (use True for FP16)
)
```

Or use the provided script:
```bash
python models/object_detector_onnx.py --convert --yolo-model yolov8n.pt --onnx-model yolov8n.onnx
```

### Step 3: Use ONNX Runtime for Inference

```python
from models.object_detector_onnx import ObjectDetectorONNX

# Initialize ONNX detector
detector = ObjectDetectorONNX(
    onnx_model_path="yolov8n.onnx",
    verbose=True,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # Try GPU first
)

# Detect objects
detections = detector.detect_objects(frame)
```

### ONNX Runtime Execution Providers

Execution providers determine which hardware/backend to use:

1. **CPUExecutionProvider** (Default)
   - Works on all systems
   - Optimized CPU kernels
   - ~1.4x faster than PyTorch CPU

2. **CUDAExecutionProvider** (GPU)
   - Requires NVIDIA GPU and CUDA
   - ~4x faster than PyTorch CPU
   - Best for most GPU systems

3. **TensorrtExecutionProvider** (TensorRT)
   - Requires NVIDIA GPU and TensorRT
   - ~5-9x faster than PyTorch CPU
   - Requires additional setup

4. **OpenVINOExecutionProvider** (Intel)
   - Optimized for Intel CPUs/GPUs
   - Good for Intel hardware

## TensorRT Implementation

TensorRT is NVIDIA's high-performance inference library. It provides the best performance but requires more setup.

### Step 1: Install TensorRT

```bash
# Download TensorRT from NVIDIA (requires NVIDIA account)
# https://developer.nvidia.com/tensorrt

# Install Python package
pip install nvidia-tensorrt

# Or use TensorRT through ONNX Runtime
pip install onnxruntime-gpu tensorrt
```

### Step 2: Convert YOLO to TensorRT

**Option A: Through ONNX (Recommended)**
```python
# First convert to ONNX
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format='onnx', imgsz=640)

# Then use TensorRT execution provider in ONNX Runtime
from models.object_detector_onnx import ObjectDetectorONNX
detector = ObjectDetectorONNX(
    onnx_model_path="yolov8n.onnx",
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
)
```

**Option B: Direct TensorRT Export (Ultralytics)**
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(
    format='engine',  # TensorRT engine format
    imgsz=640,
    half=True,        # FP16 precision
    workspace=4       # GPU memory in GB
)
```

### Step 3: Use TensorRT Engine

```python
from ultralytics import YOLO

# Load TensorRT engine
model = YOLO("yolov8n.engine")

# Use normally - TensorRT is automatically used
results = model(frame)
```

## Integration with ObjectDetector

### Option 1: Replace ObjectDetector with ONNX Version

```python
# In fem_processing.py or pipeline.py
from models.object_detector_onnx import ObjectDetectorONNX

# Instead of:
# self.object_detector = ObjectDetector(model_path=model_path)

# Use:
self.object_detector = ObjectDetectorONNX(
    onnx_model_path="yolov8n.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### Option 2: Add ONNX Support to Existing ObjectDetector

Modify `ObjectDetector` to support both PyTorch and ONNX backends:

```python
class ObjectDetector:
    def __init__(self, model_path: str, use_onnx: bool = False, ...):
        if use_onnx:
            from models.object_detector_onnx import ObjectDetectorONNX
            self.detector = ObjectDetectorONNX(model_path, ...)
        else:
            from ultralytics import YOLO
            self.detector = YOLO(model_path)
    
    def detect_objects(self, frame):
        if isinstance(self.detector, ObjectDetectorONNX):
            return self.detector.detect_objects(frame)
        else:
            # Original PyTorch implementation
            ...
```

## Benchmarking

Create a benchmark script to compare performance:

```python
import time
import numpy as np
from models.object_detector import ObjectDetector
from models.object_detector_onnx import ObjectDetectorONNX

# Test PyTorch
pytorch_detector = ObjectDetector("yolov8n.pt")
pytorch_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = pytorch_detector.detect_objects(frame)
    pytorch_times.append(time.perf_counter() - start)

# Test ONNX
onnx_detector = ObjectDetectorONNX("yolov8n.onnx")
onnx_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = onnx_detector.detect_objects(frame)
    onnx_times.append(time.perf_counter() - start)

print(f"PyTorch: {np.mean(pytorch_times)*1000:.2f}ms")
print(f"ONNX: {np.mean(onnx_times)*1000:.2f}ms")
print(f"Speedup: {np.mean(pytorch_times)/np.mean(onnx_times):.2f}x")
```

## Optimization Tips

### 1. Use FP16 (Half Precision)
```python
# When exporting to ONNX
model.export(format='onnx', half=True)

# Reduces model size and increases speed
# ~1.5-2x faster with minimal accuracy loss
```

### 2. Use INT8 Quantization (Advanced)
```python
# Requires calibration dataset
# ~2-3x faster, some accuracy loss
# Best for production deployments
```

### 3. Optimize Input Size
```python
# Smaller input = faster inference
# 320x320: ~2x faster than 640x640
# 416x416: ~1.5x faster than 640x640
model.export(format='onnx', imgsz=320)
```

### 4. Enable Graph Optimizations
```python
# ONNX Runtime automatically optimizes graphs
# Ensure simplify=True when exporting
model.export(format='onnx', simplify=True)
```

## Troubleshooting

### Issue: ONNX Runtime not using GPU
**Solution**: Check CUDA installation and ONNX Runtime GPU package
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Issue: TensorRT not available
**Solution**: Install TensorRT separately or use CUDAExecutionProvider
```bash
# TensorRT requires separate installation from NVIDIA
# ONNX Runtime with CUDAExecutionProvider is easier and still fast
```

### Issue: Model conversion fails
**Solution**: Use compatible opset version
```python
model.export(format='onnx', opset=12)  # opset 12 is widely supported
```

## Recommended Workflow

1. **Development**: Use PyTorch (easiest, good for debugging)
2. **Testing**: Convert to ONNX with CPUExecutionProvider (portable)
3. **Production (GPU)**: Use ONNX Runtime with CUDAExecutionProvider (good balance)
4. **Production (NVIDIA GPU)**: Use TensorRT (best performance)

## Files

- `models/object_detector_onnx.py` - ONNX Runtime implementation
- `TENSORRT_ONNX_GUIDE.md` - This guide
- Conversion scripts and examples included in the ONNX module

## Next Steps

1. Convert your YOLO model to ONNX
2. Test ONNX Runtime performance
3. Integrate into your pipeline
4. Consider TensorRT for maximum performance (NVIDIA GPUs only)

