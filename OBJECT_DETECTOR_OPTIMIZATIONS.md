# ObjectDetector.detect_objects() Optimization Recommendations

## Profiling Results

### Current Performance
- **Average inference time**: ~87ms per frame (11.4 FPS)
- **Bottleneck**: ~98% of time spent in YOLO model inference (PyTorch conv2d operations)
- **Post-processing**: ~2% of time (tensor extraction, dict creation, sorting)

### Key Findings
1. **Model inference dominates**: The neural network forward pass takes ~85ms
2. **Post-processing is relatively fast**: ~2ms for extracting and formatting results
3. **CPU-GPU transfers**: Multiple individual `.cpu().numpy()` calls are inefficient

## Optimizations Applied

### 1. Batch Tensor Extraction ✅
**Before:**
```python
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Individual CPU-GPU transfer
    confidence = box.conf[0].cpu().numpy()      # Individual CPU-GPU transfer
    class_id = int(box.cls[0].cpu().numpy())     # Individual CPU-GPU transfer
```

**After:**
```python
xyxy = boxes.xyxy.cpu().numpy()        # Single batch transfer
confidences = boxes.conf.cpu().numpy() # Single batch transfer
class_ids = boxes.cls.cpu().int().cpu().numpy()  # Single batch transfer
```

**Impact**: Reduces CPU-GPU transfer overhead from N transfers to 3 transfers.

### 2. Vectorized Sorting ✅
**Before:**
```python
# Create all dicts first, then sort
detections.sort(key=lambda x: x['confidence'], reverse=True)
return detections[:self.max_objects]
```

**After:**
```python
# Sort indices first, then only create dicts for top-N
top_indices = np.argsort(confidences)[::-1][:self.max_objects]
xyxy = xyxy[top_indices]
confidences = confidences[top_indices]
class_ids = class_ids[top_indices]
```

**Impact**: Avoids creating dicts for detections that will be discarded.

### 3. Cached Class Names ✅
**Before:**
```python
class_name = self.model.names[class_id]  # Dict lookup every time
```

**After:**
```python
class_names = self.model.names  # Cache once
class_name = class_names[int(class_id)]  # Use cached reference
```

**Impact**: Reduces dict attribute access overhead.

## Additional Optimization Recommendations

### 4. Frame Resizing (Optional)
For frames larger than 640px, resize before inference:
```python
if max(h, w) > 640:
    scale = 640 / max(h, w)
    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    # Scale bboxes back after detection
```
**Expected speedup**: 2-4x for large frames (1920x1080 → 640x360)

### 5. Half Precision (FP16) - GPU Only
If GPU is available, use FP16 for faster inference:
```python
if torch.cuda.is_available():
    model.model = model.model.half()
```
**Expected speedup**: 1.5-2x on modern GPUs

### 6. TensorRT/ONNX Runtime
Convert YOLO model to TensorRT or ONNX for optimized inference:
- **TensorRT**: 2-5x speedup on NVIDIA GPUs
- **ONNX Runtime**: 1.5-3x speedup with optimized execution providers

### 7. Batch Processing
If processing multiple frames, batch them together:
```python
# Process 4 frames at once
results = model([frame1, frame2, frame3, frame4])
```
**Expected speedup**: 1.5-2x for batch size 4-8

### 8. Model Quantization
Use INT8 quantization for faster inference:
```python
model.export(format='onnx', int8=True)
```
**Expected speedup**: 2-3x with minimal accuracy loss

### 9. Use Smaller Model
Consider using YOLOv8n (nano) instead of larger variants:
- **YOLOv8n**: Fastest, lower accuracy
- **YOLOv8s**: Balanced
- **YOLOv8m/l/x**: Slower, higher accuracy

### 10. Skip Frames (Frame Decimation)
Process every Nth frame instead of every frame:
```python
if frame_number % skip_frames == 0:
    detections = detector.detect_objects(frame)
```
**Expected speedup**: Nx (e.g., skip 2 frames = 2x speedup)

## Performance Impact Summary

| Optimization | Post-Processing Speedup | Overall Speedup | Difficulty |
|--------------|-------------------------|-----------------|------------|
| Batch tensor extraction | 1.5-2x | ~1.01x | Easy ✅ |
| Vectorized sorting | 1.2-1.5x | ~1.01x | Easy ✅ |
| Cached class names | 1.1x | ~1.01x | Easy ✅ |
| Frame resizing | N/A | 2-4x (large frames) | Easy |
| Half precision (FP16) | N/A | 1.5-2x | Medium |
| TensorRT/ONNX | N/A | 2-5x | Hard |
| Batch processing | N/A | 1.5-2x | Medium |
| Model quantization | N/A | 2-3x | Medium |
| Smaller model | N/A | 2-3x | Easy |
| Frame decimation | N/A | Nx | Easy |

## Recommended Implementation Order

1. ✅ **Applied**: Batch tensor extraction, vectorized sorting, cached names
2. **Next**: Frame resizing for large inputs (easy, high impact)
3. **Then**: Half precision if GPU available (medium effort, good speedup)
4. **Advanced**: TensorRT/ONNX conversion (hard, best speedup)

## Code Example: Full Optimized Version

See `models/object_detector.py` for the optimized implementation with:
- Batch tensor operations
- Vectorized sorting
- Cached class names
- Early limiting

For additional optimizations (resizing, FP16, etc.), see `models/object_detector_optimized.py`.

