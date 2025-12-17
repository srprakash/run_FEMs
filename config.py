"""
Configuration settings for the video processing pipeline.
"""

import os
from pathlib import Path

# Model configurations
YOLO_MODEL_PATH = "yolov8n.pt"  # Will be downloaded automatically
POSE_MODEL_PATH = "pose_landmarker.task"  # MediaPipe pose model
PERSON_REID_MODEL_PATH = "person_reid_model.pth"  # Custom person re-identification model

# Processing parameters
MAX_OBJECTS_PER_FRAME = 15
MAX_PERSONS_PER_FRAME = 3
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3  # IoU threshold for YOLO NMS (lower = more aggressive deduplication)
POSE_CONFIDENCE_THRESHOLD = 0.7

# Performance optimization settings
TARGET_FPS = 5  # Process at 5 FPS instead of full video FPS
RESIZE_FRAMES = True  # Resize frames for faster processing
TARGET_WIDTH = 640  # Target width for resized frames (height calculated to maintain aspect ratio)
USE_HOLISTIC_MODEL = True  # Use MediaPipe Holistic for pose detection (faster than separate pose model)

# Output settings
OUTPUT_CSV_COLUMNS = [
    'frame_number',
    'timestamp',
    'object_class_1', 'object_confidence_1', 'object_x1_1', 'object_y1_1', 'object_x2_1', 'object_y2_1',
    'object_class_2', 'object_confidence_2', 'object_x1_2', 'object_y1_2', 'object_x2_2', 'object_y2_2',
    'object_class_3', 'object_confidence_3', 'object_x1_3', 'object_y1_3', 'object_x2_3', 'object_y2_3',
    'object_class_4', 'object_confidence_4', 'object_x1_4', 'object_y1_4', 'object_x2_4', 'object_y2_4',
    'object_class_5', 'object_confidence_5', 'object_x1_5', 'object_y1_5', 'object_x2_5', 'object_y2_5',
    'object_class_6', 'object_confidence_6', 'object_x1_6', 'object_y1_6', 'object_x2_6', 'object_y2_6',
    'object_class_7', 'object_confidence_7', 'object_x1_7', 'object_y1_7', 'object_x2_7', 'object_y2_7',
    'object_class_8', 'object_confidence_8', 'object_x1_8', 'object_y1_8', 'object_x2_8', 'object_y2_8',
    'object_class_9', 'object_confidence_9', 'object_x1_9', 'object_y1_9', 'object_x2_9', 'object_y2_9',
    'object_class_10', 'object_confidence_10', 'object_x1_10', 'object_y1_10', 'object_x2_10', 'object_y2_10',
    'object_class_11', 'object_confidence_11', 'object_x1_11', 'object_y1_11', 'object_x2_11', 'object_y2_11',
    'object_class_12', 'object_confidence_12', 'object_x1_12', 'object_y1_12', 'object_x2_12', 'object_y2_12',
    'object_class_13', 'object_confidence_13', 'object_x1_13', 'object_y1_13', 'object_x2_13', 'object_y2_13',
    'object_class_14', 'object_confidence_14', 'object_x1_14', 'object_y1_14', 'object_x2_14', 'object_y2_14',
    'object_class_15', 'object_confidence_15', 'object_x1_15', 'object_y1_15', 'object_x2_15', 'object_y2_15',
    'person_1_visible', 'person_1_centroid_x', 'person_1_centroid_y', 'person_1_pose_landmarks',
    'person_2_visible', 'person_2_centroid_x', 'person_2_centroid_y', 'person_2_pose_landmarks',
    'person_3_visible', 'person_3_centroid_x', 'person_3_centroid_y', 'person_3_pose_landmarks'
]

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

# Audio processing settings (for future implementation)
AUDIO_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 1024
