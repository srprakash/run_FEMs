# Configure YOLO NMS to Prevent Duplicate Person Detections

## Quick Fix

**File**: `run_FEMs/models/object_detector.py`, line 453

**Change**:
```python
# Before:
results = self.model(frame, conf=self.confidence_threshold, verbose=self.verbose)

# After:
results = self.model(frame, conf=self.confidence_threshold, iou=0.3, verbose=self.verbose)
```

This lowers YOLO's NMS IoU threshold from ~0.45 (default) to 0.3, making it more aggressive at removing overlapping boxes.

## Better: Make it Configurable

### 1. Add to `config.py`:

```python
# Model configurations
YOLO_MODEL_PATH = "yolov8n.pt"
POSE_MODEL_PATH = "pose_landmarker.task"
PERSON_REID_MODEL_PATH = "person_reid_model.pth"

# Detection parameters
MAX_OBJECTS_PER_FRAME = 15
MAX_PERSONS_PER_FRAME = 3
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3  # ← ADD THIS
POSE_CONFIDENCE_THRESHOLD = 0.7
```

### 2. Update `object_detector.py`:

**Import the config**:
```python
from config import MAX_OBJECTS_PER_FRAME, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD
```

**Use in detection** (line ~54):
```python
def __init__(self, model_path: str = "yolov8n.pt", verbose: bool = False, 
             force_onnx: bool = None, onnx_model_path: str = None):
    self.max_objects = MAX_OBJECTS_PER_FRAME
    self.confidence_threshold = CONFIDENCE_THRESHOLD
    self.nms_iou_threshold = NMS_IOU_THRESHOLD  # ← ADD THIS
    self.verbose = verbose
    # ... rest of __init__
```

**Use in detection** (line ~453):
```python
# Use YOLO (ultralytics) backend
results = self.model(
    frame, 
    conf=self.confidence_threshold,
    iou=self.nms_iou_threshold,  # ← ADD THIS
    verbose=self.verbose
)
```

## What is NMS IoU Threshold?

**IoU (Intersection over Union)** measures how much two bounding boxes overlap:
- 0.0 = No overlap
- 1.0 = Perfect overlap

**NMS (Non-Maximum Suppression)** removes duplicate detections:
- Keeps the box with highest confidence
- Removes boxes with IoU > threshold

### Tuning Guide

**More Aggressive (stricter deduplication)**:
```python
NMS_IOU_THRESHOLD = 0.2  # Very aggressive, may lose close persons
NMS_IOU_THRESHOLD = 0.3  # Aggressive (recommended for stationary scenes)
```

**Balanced**:
```python
NMS_IOU_THRESHOLD = 0.4  # Good for most cases
NMS_IOU_THRESHOLD = 0.45 # YOLO default
```

**More Lenient** (allows more overlap):
```python
NMS_IOU_THRESHOLD = 0.5  # May still have some duplicates
NMS_IOU_THRESHOLD = 0.6  # Lenient
```

## Expected Results

### Before (default IoU ~0.45):
```
[ObjectDetector] Found 3 bounding boxes for same person
    Box 1: [100, 100, 200, 300], conf=0.92
    Box 2: [105, 95, 205, 295], conf=0.88, IoU with Box 1 = 0.85
    Box 3: [102, 98, 198, 298], conf=0.91, IoU with Box 1 = 0.90
→ All 3 boxes pass through YOLO NMS (IoU < 0.45? No, but default is lenient)
```

### After (IoU = 0.3):
```
[ObjectDetector] Found 1 bounding box
    Box 1: [100, 100, 200, 300], conf=0.92
→ Box 2 and Box 3 removed by YOLO NMS (IoU > 0.3)
```

## Testing

```bash
cd /Users/srprakash/DistalCare/FEM/run_FEMs

# Edit models/object_detector.py (add iou=0.3 to line 453)

# Test detection
python main.py --video data/local/Activity_1Person_001.mov --visualize

# Should see:
# - Fewer duplicate bounding boxes on same person
# - Cleaner detection output
```

## Comparison: YOLO NMS vs DCHub NMS

|  | YOLO NMS (run_FEMs) | DCHub NMS (fem_processing.py) |
|---|---|---|
| **When** | During YOLO inference | After person detection |
| **Scope** | All objects (persons, cars, etc.) | Only persons |
| **Efficiency** | Better (prevents duplicates at source) | Less efficient (processes duplicates first) |
| **Flexibility** | Less flexible (global setting) | More flexible (per-person filtering) |
| **Recommended** | **Primary solution** | Fallback/safety net |

## Recommendation

**Use both**:
1. **Set YOLO NMS IoU to 0.3-0.4** (primary fix, in run_FEMs)
2. **Keep DCHub NMS at 0.5** (safety net, in DCHub/config.json)

This provides defense-in-depth:
- YOLO filters most duplicates (0.3 threshold)
- DCHub catches any that slip through (0.5 threshold)
- Minimal performance impact (DCHub NMS rarely triggers)

## Additional YOLO Parameters

You can also pass other parameters to tune detection:

```python
results = self.model(
    frame,
    conf=0.5,        # Confidence threshold
    iou=0.3,         # NMS IoU threshold
    max_det=50,      # Max detections per image
    agnostic_nms=False,  # Class-agnostic NMS (True = merge across classes)
    classes=[0],     # Only detect specific classes (0 = person)
)
```

For your use case (person tracking), you might also want:
```python
results = self.model(
    frame,
    conf=0.5,
    iou=0.3,
    classes=[0],  # Only detect persons, ignore other objects
)
```

