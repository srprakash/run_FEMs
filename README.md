# run_FEMs

A comprehensive computer vision pipeline for analyzing videos using multiple AI models. This project detects objects, tracks people across frames, and extracts pose information, exporting all results to structured CSV format.

## Features

- **Dual Detection Modes**: 
  - **Holistic Mode**: MediaPipe Holistic for combined pose + person detection (faster)
  - **Separate Models**: YOLOv8 + MediaPipe Pose for maximum accuracy
- **Performance Optimizations**:
  - **Frame Decimation**: Process at 5 FPS instead of full video FPS
  - **Frame Resizing**: Automatically resize frames to 640px width for faster processing
  - **Single Model Processing**: Holistic mode eliminates redundant model calls
- **Person Tracking**: Feature-based re-identification for consistent person tracking across frames
- **Pose Detection**: MediaPipe pose detection with 33 landmark points per person
- **Real-time Visualization**: Display video frames with bounding boxes, labels, and pose keypoints
- **Video Export**: Save annotated videos with all detections and overlays
- **CSV Export**: Structured data export with comprehensive detection information
- **Batch Processing**: Process single videos or entire folders
- **Configurable**: Adjustable confidence thresholds and processing parameters

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- MediaPipe
- Ultralytics (YOLO)

### Install Dependencies

```bash
pip install opencv-python
pip install ultralytics
pip install mediapipe
pip install pandas
pip install numpy
pip install scikit-learn
pip install librosa
pip install soundfile
pip install tqdm
```

## Project Structure

```
run_FEMs/
├── __init__.py
├── config.py                 # Configuration settings
├── pipeline.py              # Main processing pipeline
├── video_processor.py       # Video frame extraction
├── main.py                  # Entry point script
├── models/
│   ├── __init__.py
│   ├── object_detector.py   # YOLO object detection
│   ├── person_detector.py   # Person tracking & re-identification
│   └── pose_detector.py     # MediaPipe pose detection
├── data_export/
│   ├── __init__.py
│   └── csv_exporter.py      # CSV data export
└── utils/
    └── __init__.py
```

## Usage

### Command Line Interface

```bash
# Process a single video
python main.py --input path/to/video.mp4

# Process with real-time visualization
python main.py --input path/to/video.mp4 --visualize

# Process and save annotated video
python main.py --input path/to/video.mp4 --save-video

# Process with both visualization and save annotated video
python main.py --input path/to/video.mp4 --visualize --save-video

# Process a single video with custom output name
python main.py --input path/to/video.mp4 --output my_results.csv

# Process all videos in a folder
python main.py --input path/to/video/folder --batch

# Process with custom output directory
python main.py --input path/to/video.mp4 --output-dir results/
```

### Programmatic Usage

```python
from pipeline import VideoProcessingPipeline

# Initialize pipeline
pipeline = VideoProcessingPipeline(output_dir="output")

# Process single video with visualization
result = pipeline.process_video("path/to/video.mp4", visualize=True)
print(f"Processing completed: {result}")

# Process single video and save annotated video
result = pipeline.process_video("path/to/video.mp4", save_video=True)

# Process folder of videos
results = pipeline.process_folder("path/to/video/folder")
for result in results:
    if result['success']:
        print(f"✓ {result['video_file']}")
    else:
        print(f"✗ {result['video_file']}: {result['error']}")

# Cleanup
pipeline.cleanup()
```

## Output Format

The pipeline exports CSV files with the following structure:

### Object Detection Data (per frame)
- `frame_number`: Frame index
- `timestamp`: Time in seconds
- `object_class_1` to `object_class_15`: Detected object classes
- `object_confidence_1` to `object_confidence_15`: Detection confidence scores
- `object_x1_1` to `object_x1_15`: Bounding box top-left X coordinates
- `object_y1_1` to `object_y1_15`: Bounding box top-left Y coordinates
- `object_x2_1` to `object_x2_15`: Bounding box bottom-right X coordinates
- `object_y2_1` to `object_y2_15`: Bounding box bottom-right Y coordinates

### Person Tracking Data (per frame)
- `person_1_visible` to `person_3_visible`: Person visibility flags
- `person_1_centroid_x` to `person_3_centroid_x`: Person centroid X coordinates
- `person_1_centroid_y` to `person_3_centroid_y`: Person centroid Y coordinates
- `person_1_pose_landmarks` to `person_3_pose_landmarks`: Pose landmark coordinates

## Configuration

Edit `config.py` to adjust processing parameters:

```python
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
POSE_CONFIDENCE_THRESHOLD = 0.7

# Processing limits
MAX_OBJECTS_PER_FRAME = 15
MAX_PERSONS_PER_FRAME = 3

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
```

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)

## Performance

### Optimized Performance
With the new optimizations, the pipeline can process videos at **real-time or faster** speeds:

- **Holistic Mode**: 15-30+ FPS (recommended for speed)
- **Separate Models**: 5-15 FPS (maximum accuracy)
- **Frame Decimation**: Processes at 5 FPS regardless of input video FPS
- **Frame Resizing**: Automatically reduces processing load by resizing to 640px width

### Performance Factors
Processing speed depends on:
- **Detection Mode**: Holistic vs Separate Models
- **Video Resolution**: Automatically resized for optimal performance
- **Target FPS**: Configurable processing rate (default: 5 FPS)
- **Hardware**: CPU/GPU specifications

### Configuration
Adjust performance settings in `config.py`:
```python
TARGET_FPS = 5  # Process at 5 FPS
RESIZE_FRAMES = True  # Enable frame resizing
TARGET_WIDTH = 640  # Target width for resized frames
USE_HOLISTIC_MODEL = True  # Use faster holistic mode
```

## Model Information

### Holistic Mode (Default - Faster)
- **Detection**: MediaPipe Holistic - combined pose detection and person bounding box extraction
- **Performance**: 15-30+ FPS
- **Use Case**: Real-time processing, general analysis

### Separate Models Mode (Maximum Accuracy)
- **Object Detection**: YOLOv8n (nano) - lightweight and fast
- **Person Re-identification**: Custom feature-based tracking using histogram features
- **Pose Detection**: MediaPipe Pose - 33 landmark points per person
- **Performance**: 5-15 FPS
- **Use Case**: Maximum accuracy, detailed analysis

## Visualization

The pipeline includes comprehensive visualization capabilities:

### Real-time Display
- **Bounding Boxes**: Color-coded boxes for different object classes
  - Green: Person
  - Blue: Chair  
  - Red: Other objects
- **Confidence Scores**: Displayed above each detection
- **Person IDs**: Tracked person identifiers
- **Pose Landmarks**: 33-point skeleton overlay with connections

### Video Export
- Save annotated videos with all visualizations
- Maintains original video quality and frame rate
- Perfect for presentations and analysis

### Interactive Controls
- Press 'q' during visualization to quit processing
- Real-time frame-by-frame analysis
- Immediate feedback on detection quality

## Examples

### Basic Video Analysis
```bash
python main.py --input sample_video.mp4
```

### Real-time Visualization
```bash
python main.py --input sample_video.mp4 --visualize
```

### Save Annotated Video
```bash
python main.py --input sample_video.mp4 --save-video
```

### Batch Processing
```bash
python main.py --input videos/ --batch --output-dir results/
```

### High Confidence Processing
Modify `config.py`:
```python
CONFIDENCE_THRESHOLD = 0.8
POSE_CONFIDENCE_THRESHOLD = 0.9
```

## Troubleshooting

### Common Issues

1. **Model Download**: YOLO models are downloaded automatically on first use
2. **Memory Usage**: Large videos may require significant RAM
3. **Processing Speed**: Consider reducing video resolution for faster processing
4. **Dependencies**: Ensure all required packages are installed

### Error Messages

- `Video file not found`: Check file path and permissions
- `Unsupported video format`: Use supported formats listed above
- `Could not open video file`: Video file may be corrupted

## License

This project is open source. Please check individual model licenses:
- YOLOv8: AGPL-3.0
- MediaPipe: Apache 2.0

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.
