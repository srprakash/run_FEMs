# ONNX Runtime CPU Optimization Guide

## Overview

This project now automatically uses ONNX Runtime for object detection on non-GPU platforms (Intel Mac and Raspberry Pi). This provides better performance and lower memory usage compared to running PyTorch-based YOLO models on CPU.

## Platform Detection

The system automatically detects the platform and selects the appropriate backend:

- **Intel Mac**: Uses ONNX Runtime (CPU)
- **Raspberry Pi**: Uses ONNX Runtime (CPU)
- **Other platforms**: Uses YOLO (ultralytics) with GPU if available, CPU otherwise

## Quantized Models and CPU Speedup

### What is Model Quantization?

Model quantization reduces the precision of model weights and activations from 32-bit floating point (FP32) to lower precision formats like:
- **INT8 (8-bit integers)**: 4x smaller model size, typically 2-4x faster inference
- **FP16 (16-bit floating point)**: 2x smaller model size, typically 1.5-2x faster inference

### Expected Speedup with Quantized Models on CPU

**Yes, quantized models (especially INT8) provide significant speedup on CPU:**

1. **INT8 Quantization Benefits:**
   - **2-4x faster inference** compared to FP32
   - **4x smaller model size** (faster loading, less memory)
   - **Better cache utilization** (more data fits in CPU cache)
   - **Optimized CPU instructions** (many CPUs have INT8 SIMD instructions)
   - **Lower memory bandwidth** requirements

2. **Typical Performance Gains:**
   - **FP32 → INT8**: 2-4x speedup (most common)
   - **FP32 → FP16**: 1.5-2x speedup
   - **Accuracy loss**: Usually < 1-2% for INT8, negligible for FP16

3. **Why CPU Benefits More:**
   - CPUs have optimized INT8 instructions (AVX-512, NEON on ARM)
   - Memory bandwidth is often the bottleneck on CPU; smaller models help
   - Better cache locality with smaller data types
   - ONNX Runtime has highly optimized INT8 kernels for CPU

### Converting to Quantized ONNX Models

To create a quantized ONNX model, you can use ONNX Runtime's quantization tools:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization (recommended for CPU)
quantize_dynamic(
    model_input="yolov8n.onnx",
    model_output="yolov8n_int8.onnx",
    weight_type=QuantType.QUInt8  # or QuantType.QInt8
)
```

Or use the ONNX Runtime quantization tool:

```bash
pip install onnxruntime-tools
python -m onnxruntime.quantization.preprocess --input yolov8n.onnx --output yolov8n_preprocessed.onnx
python -m onnxruntime.quantization.quantize --input yolov8n_preprocessed.onnx --output yolov8n_int8.onnx --quantization_mode IntegerOps
```

### Using Quantized Models

The `ObjectDetector` class will automatically use an ONNX model if found:

1. **Automatic Detection**: If `yolov8n.onnx` exists, it will be used on non-GPU platforms
2. **Manual Override**: You can specify the ONNX model path:
   ```python
   detector = ObjectDetector(
       model_path="yolov8n.pt",  # Fallback
       onnx_model_path="yolov8n_int8.onnx",  # Quantized model
       verbose=True
   )
   ```

### Performance Comparison

Expected inference times on CPU (640x640 input):

| Model Type | Intel Mac (CPU) | Raspberry Pi 4 (CPU) |
|------------|----------------|---------------------|
| YOLO FP32 (PyTorch) | ~200-300ms | ~800-1200ms |
| ONNX FP32 | ~150-200ms | ~600-900ms |
| ONNX INT8 | ~50-100ms | ~200-400ms |

*Note: Actual performance depends on CPU model, model size, and input resolution.*

## Current Implementation

The `ObjectDetector` class in `models/object_detector.py`:

1. **Auto-detects platform** using `utils/platform_detection.py`
2. **Automatically uses ONNX Runtime** on Intel Mac and Raspberry Pi
3. **Falls back to YOLO** if ONNX Runtime is not available
4. **Supports both backends** with the same API

## Migration Notes

- **No code changes required**: The API remains the same
- **Model files**: Ensure `yolov8n.onnx` exists in the project directory
- **Dependencies**: Install `onnxruntime` for CPU: `pip install onnxruntime`
- **Quantized models**: Optional but recommended for best performance

## Recommendations

1. **For Intel Mac**: Use ONNX FP32 or INT8 models for best performance
2. **For Raspberry Pi**: Use ONNX INT8 models for acceptable real-time performance
3. **For GPU platforms**: YOLO (ultralytics) with GPU is still fastest
4. **Model conversion**: Convert YOLO models to ONNX using the provided utilities

## Example Usage

```python
from models.object_detector import ObjectDetector
import cv2

# Automatic: Uses ONNX on non-GPU platforms, YOLO otherwise
detector = ObjectDetector(verbose=True)

# Process a frame
frame = cv2.imread("test.jpg")
detections = detector.detect_objects(frame)

for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

## Troubleshooting

1. **ONNX model not found**: Ensure `yolov8n.onnx` exists or provide `onnx_model_path`
2. **Import errors**: Install ONNX Runtime: `pip install onnxruntime`
3. **Performance issues**: Try using a quantized INT8 model
4. **Accuracy concerns**: INT8 quantization typically has < 1-2% accuracy loss

