"""
Video processing module for extracting frames and audio buffers.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import librosa
import soundfile as sf
from config import (SUPPORTED_VIDEO_FORMATS, AUDIO_SAMPLE_RATE, AUDIO_BUFFER_SIZE,
                    RESIZE_FRAMES, TARGET_WIDTH, TARGET_FPS)


class VideoProcessor:
    """Handles video frame extraction and audio buffer capture."""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.duration = 0
        self.width = 0
        self.height = 0
        self.resize_frames = RESIZE_FRAMES
        self.target_width = TARGET_WIDTH
        self.target_fps = TARGET_FPS
        self.frame_skip = 1  # Will be calculated based on target FPS
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self):
        """Open video file for processing."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        if self.video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"Unsupported video format: {self.video_path.suffix}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame skip for target FPS
        if self.fps > 0 and self.target_fps > 0:
            self.frame_skip = max(1, int(self.fps / self.target_fps))
        else:
            self.frame_skip = 1
            
        # Calculate resize dimensions if needed
        if self.resize_frames and self.target_width > 0:
            aspect_ratio = self.height / self.width
            self.target_height = int(self.target_width * aspect_ratio)
        else:
            self.target_width = self.width
            self.target_height = self.height
        
    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by frame number.
        
        Args:
            frame_number: Frame number to extract
            
        Returns:
            Frame as numpy array or None if not found
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Video not opened. Call open() first.")
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return frame
        return None
        
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array or None if not found
        """
        frame_number = int(timestamp * self.fps)
        return self.get_frame(frame_number)
        
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame if needed.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        if self.resize_frames and (frame.shape[1] != self.target_width or frame.shape[0] != self.target_height):
            return cv2.resize(frame, (self.target_width, self.target_height))
        return frame
        
    def iterate_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate through frames in the video with decimation and resizing.
        
        Yields:
            Tuple of (frame_number, frame_array)
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Video not opened. Call open() first.")
            
        frame_number = 0
        processed_frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Skip frames based on target FPS
            if frame_number % self.frame_skip == 0:
                # Resize frame if needed
                processed_frame = self._resize_frame(frame)
                yield processed_frame_count, processed_frame
                processed_frame_count += 1
                
            frame_number += 1
            
    def extract_audio_buffer(self, start_time: float, duration: float) -> Optional[np.ndarray]:
        """
        Extract audio buffer for a specific time range.
        
        Args:
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Audio buffer as numpy array or None if extraction fails
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(str(self.video_path), 
                                        offset=start_time, 
                                        duration=duration,
                                        sr=AUDIO_SAMPLE_RATE)
            return audio_data
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
            
    def get_video_info(self) -> dict:
        """
        Get video information.
        
        Returns:
            Dictionary with video properties
        """
        return {
            'path': str(self.video_path),
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'width': self.width,
            'height': self.height,
            'format': self.video_path.suffix,
            'processed_width': self.target_width,
            'processed_height': self.target_height,
            'target_fps': self.target_fps,
            'frame_skip': self.frame_skip
        }
