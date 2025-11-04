"""
Main video processing pipeline orchestrator.
"""

import os
import time
import cv2
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from video_processor import VideoProcessor
from models.object_detector import ObjectDetector
from models.person_detector import PersonDetector
from models.pose_detector import PoseDetector
from models.holistic_detector import HolisticDetector
from data_export.csv_exporter import CSVExporter
from config import SUPPORTED_VIDEO_FORMATS, USE_HOLISTIC_MODEL


class VideoProcessingPipeline:
    """Main pipeline for processing videos with computer vision models."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the processing pipeline.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models - always use YOLO for all object detection
        self.object_detector = ObjectDetector()
        self.person_detector = PersonDetector()
        
        # Use holistic model for pose detection if enabled, otherwise use separate pose detector
        self.use_holistic_pose = USE_HOLISTIC_MODEL
        if self.use_holistic_pose:
            self.holistic_detector = HolisticDetector()
            print("Using MediaPipe Holistic for pose detection on detected persons")
        else:
            self.pose_detector = PoseDetector()
            print("Using MediaPipe Pose for pose detection")
            
        self.csv_exporter = CSVExporter()
        
    def process_video(self, video_path: str, output_filename: str = None, 
                     visualize: bool = False, save_video: bool = False) -> Dict[str, Any]:
        """
        Process a single video file.
        
        Args:
            video_path: Path to the video file
            output_filename: Name for output CSV file (optional)
            visualize: Whether to display video frames with overlays
            save_video: Whether to save annotated video
            
        Returns:
            Dictionary with processing results and statistics
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")
            
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = f"{video_path.stem}_processed.csv"
            
        output_path = self.output_dir / output_filename
        
        print(f"Processing video: {video_path}")
        print(f"Output will be saved to: {output_path}")
        
        start_time = time.time()
        
        # Process video
        with VideoProcessor(str(video_path)) as video_processor:
            video_info = video_processor.get_video_info()
            print(f"Video info: {video_info}")
            
            # Setup video writer if saving video
            video_writer = None
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(self.output_dir / f"{video_path.stem}_annotated.mp4"),
                    fourcc, video_info['target_fps'], (video_info['processed_width'], video_info['processed_height'])
                )
            
            # Process each frame
            frame_count = 0
            for frame_number, frame in tqdm(video_processor.iterate_frames(), 
                                          desc="Processing frames"):
                # Calculate timestamp based on processed frame rate
                timestamp = frame_number / video_info['target_fps']
                
                # Step 1: Detect all objects using YOLO (persons, chairs, etc.)
                object_detections = self.object_detector.detect_objects(frame)
                
                # Step 2: Extract person detections and track them
                person_detections = self.person_detector.detect_persons(frame, object_detections)
                tracked_persons = self.person_detector.track_persons(frame, person_detections)
                
                # Step 3: Detect poses for tracked persons
                if self.use_holistic_pose:
                    # Use holistic model for pose detection
                    person_bboxes = [person['bbox'] for person in tracked_persons]
                    holistic_pose_detections = self.holistic_detector.detect_poses_for_persons(frame, person_bboxes)
                    
                    # Match holistic poses to tracked persons (already aligned by index)
                    pose_detections = []
                    for i, person in enumerate(tracked_persons):
                        pose_detection = {
                            'person_id': person['person_id'],
                            'bbox': person['bbox'],
                            'landmarks': None,
                            'visibility': False,
                            'landmark_coords': None
                        }
                        
                        # Holistic detector returns one detection per person bbox
                        if i < len(holistic_pose_detections):
                            holistic_pose = holistic_pose_detections[i]
                            if holistic_pose['visibility']:
                                pose_detection['landmarks'] = holistic_pose['landmarks']
                                pose_detection['visibility'] = True
                                pose_detection['landmark_coords'] = holistic_pose['landmark_coords']
                        
                        pose_detections.append(pose_detection)
                else:
                    # Use separate pose detector
                    pose_detections = self.pose_detector.detect_poses(frame, tracked_persons)
                
                # Add to CSV export
                self.csv_exporter.add_frame_data(
                    frame_number, timestamp, object_detections, pose_detections
                )
                
                # Visualization
                if visualize or save_video:
                    vis_frame = self._create_visualization_frame(
                        frame, object_detections, tracked_persons, pose_detections
                    )
                    
                    if visualize:
                        cv2.imshow('Video Processing Pipeline', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Processing stopped by user")
                            break
                    
                    if save_video and video_writer:
                        video_writer.write(vis_frame)
                
                frame_count += 1
                
        # Cleanup
        if video_writer:
            video_writer.release()
        if visualize:
            cv2.destroyAllWindows()
        
        # Export results
        self.csv_exporter.export_to_csv(str(output_path))
        
        # Get processing statistics
        processing_time = time.time() - start_time
        stats = self.csv_exporter.get_summary_stats()
        stats.update({
            'processing_time': processing_time,
            'fps_processed': frame_count / processing_time if processing_time > 0 else 0,
            'video_info': video_info
        })
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Processed {frame_count} frames at {stats['fps_processed']:.2f} FPS")
        print(f"Summary: {stats}")
        
        return stats
        
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process all videos in a folder.
        
        Args:
            folder_path: Path to folder containing videos
            
        Returns:
            List of processing results for each video
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        # Find all video files
        video_files = []
        for ext in SUPPORTED_VIDEO_FORMATS:
            video_files.extend(folder_path.glob(f"*{ext}"))
            video_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
        if not video_files:
            print(f"No video files found in {folder_path}")
            return []
            
        print(f"Found {len(video_files)} video files to process")
        
        results = []
        for video_file in video_files:
            try:
                print(f"\nProcessing: {video_file.name}")
                result = self.process_video(str(video_file))
                results.append({
                    'video_file': str(video_file),
                    'success': True,
                    'result': result
                })
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                results.append({
                    'video_file': str(video_file),
                    'success': False,
                    'error': str(e)
                })
                
        return results
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'object_detector': 'YOLOv8',
            'person_detector': 'Custom feature-based re-identification with EMA and spatial matching',
            'pose_detector': 'MediaPipe Holistic' if self.use_holistic_pose else 'MediaPipe Pose',
            'models_loaded': True
        }
        return info
        
    def _create_visualization_frame(self, frame, object_detections, tracked_persons, pose_detections):
        """
        Create a visualization frame with all detections and annotations.
        
        Args:
            frame: Original frame
            object_detections: List of object detections
            tracked_persons: List of tracked persons
            pose_detections: List of pose detections
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw object detections
        for detection in object_detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on class
            if class_name == 'person':
                color = (0, 255, 0)  # Green for person
            elif class_name == 'chair':
                color = (255, 0, 0)  # Blue for chair
            else:
                color = (0, 0, 255)  # Red for other objects
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw person tracking IDs
        for person in tracked_persons:
            x1, y1, x2, y2 = map(int, person['bbox'])
            person_id = person['person_id']
            
            # Draw person ID
            cv2.putText(vis_frame, f"ID: {person_id}", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw pose landmarks
        for pose_detection in pose_detections:
            if pose_detection['landmarks']:
                # Use MediaPipe's built-in drawing
                if self.use_holistic_pose:
                    # Use holistic drawing
                    self.holistic_detector.mp_drawing.draw_landmarks(
                        vis_frame,
                        pose_detection['landmarks'],
                        self.holistic_detector.mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.holistic_detector.mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=self.holistic_detector.mp_drawing.DrawingSpec(
                            color=(255, 0, 255), thickness=2
                        )
                    )
                else:
                    # Use pose detector drawing
                    self.pose_detector.mp_drawing.draw_landmarks(
                        vis_frame,
                        pose_detection['landmarks'],
                        self.pose_detector.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.pose_detector.mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=self.pose_detector.mp_drawing.DrawingSpec(
                            color=(255, 0, 255), thickness=2
                        )
                    )
        
        return vis_frame
        
    def cleanup(self):
        """Cleanup resources."""
        if self.use_holistic_pose:
            if hasattr(self.holistic_detector, 'holistic'):
                self.holistic_detector.holistic.close()
        else:
            if hasattr(self.pose_detector, 'pose'):
                self.pose_detector.pose.close()
        self.csv_exporter.clear_data()
