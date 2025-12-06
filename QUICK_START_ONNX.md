# Quick Start: ONNX Runtime Optimization

## 5-Minute Setup

### Step 1: Install ONNX Runtime

```bash
# For CPU
pip install onnxruntime

# For GPU (NVIDIA)
pip install onnxruntime-gpu
```

### Step 2: Convert YOLO Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format='onnx', imgsz=640, simplify=True)
# Creates: yolov8n.onnx
```

Or use the provided script:
```bash
python models/object_detector_onnx.py --convert --yolo-model yolov8n.pt --onnx-model yolov8n.onnx
```

### Step 3: Use ONNX Runtime

```python
from models.object_detector_onnx import ObjectDetectorONNX

# Initialize
detector = ObjectDetectorONNX("yolov8n.onnx")

# Use like normal
detections = detector.detect_objects(frame)
```

### Step 4: Integrate into Your Code

In `fem_processing.py`, replace:
```python
self.object_detector = ObjectDetector(model_path=model_path, verbose=False)
```

With:
```python
from models.object_detector_onnx import ObjectDetectorONNX
self.object_detector = ObjectDetectorONNX(
    onnx_model_path=str(_run_fems_dir / "yolov8n.onnx"),
    verbose=False
)
```

## Expected Performance

- **CPU**: ~1.4x faster than PyTorch
- **GPU**: ~4x faster than PyTorch CPU
- **TensorRT**: ~5-9x faster (NVIDIA GPUs only)

## Benchmark

```bash
python benchmark_onnx.py
```

## Troubleshooting

**Q: ONNX Runtime not using GPU?**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Q: Model conversion fails?**
- Ensure `ultralytics` is installed: `pip install ultralytics`
- Try different opset: `model.export(format='onnx', opset=11)`

**Q: Different results than PyTorch?**
- Normal - slight numerical differences are expected
- Accuracy should be very similar (<1% difference)

## Full Documentation

See `TENSORRT_ONNX_GUIDE.md` for detailed information.

