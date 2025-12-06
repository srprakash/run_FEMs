"""
Main video processing pipeline orchestrator.
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from video_processor import VideoProcessor
from models.object_detector import ObjectDetector
from models.person_detector import PersonDetector
from models.pose_detector import PoseDetector
from models.holistic_detector import HolisticDetector
from models.person_signature_manager import PersonSignatureManager
from data_export.csv_exporter import CSVExporter
from utils.person_tagging_ui import PersonTaggingUI
from config import SUPPORTED_VIDEO_FORMATS, USE_HOLISTIC_MODEL


class VideoProcessingPipeline:
    """Main pipeline for processing videos with computer vision models."""
    
    def __init__(self, output_dir: str = "output", signature_file: str = "person_signatures.pkl", 
                 enable_tagging: bool = True, verbose: bool = False):
        """
        Initialize the processing pipeline.
        
        Args:
            output_dir: Directory to save output files
            signature_file: Path to person signature file
            enable_tagging: Whether to enable person tagging functionality
            verbose: Whether to show verbose output (including initialization messages)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Initialize signature manager
        self.signature_manager = None
        self.tagging_ui = None
        self.enable_tagging = enable_tagging
        
        if enable_tagging:
            # Store signatures in 'Landmarks' directory one level above the output directory
            # Example: outputs/Video1 -> outputs/Landmarks/person_signatures.pkl
            parent_dir = self.output_dir.parent
            landmarks_dir = parent_dir / "Landmarks"
            landmarks_dir.mkdir(parents=True, exist_ok=True)
            signature_path = landmarks_dir / signature_file
            self.signature_manager = PersonSignatureManager(str(signature_path))
            self.signature_manager.load_signatures()
            self.tagging_ui = PersonTaggingUI()
            if verbose:
                print(f"Person tagging enabled. Signature file: {signature_path}")
        
        # Initialize models - always use YOLO for all object detection
        # ObjectDetector verbose will be set later based on process_video verbose parameter
        self.object_detector = ObjectDetector(verbose=False)  # Default to non-verbose, updated per video
        self.person_detector = PersonDetector(signature_manager=self.signature_manager)
        
        # Use holistic model for pose detection if enabled, otherwise use separate pose detector
        self.use_holistic_pose = USE_HOLISTIC_MODEL
        if self.use_holistic_pose:
            self.holistic_detector = HolisticDetector()
            if verbose:
                print("Using MediaPipe Holistic for pose detection on detected persons")
        else:
            self.pose_detector = PoseDetector()
            if verbose:
                print("Using MediaPipe Pose for pose detection")
            
        self.csv_exporter = CSVExporter()
        
        # Store frame samples for gallery display
        self.frame_samples = {}  # person_id -> sample frame image
        
    def process_video(self, video_path: str, output_filename: str = None, 
                     visualize: bool = False, save_video: bool = False, 
                     verbose: bool = False) -> Dict[str, Any]:
        """
        Process a single video file.
        
        Args:
            video_path: Path to the video file
            output_filename: Name for output CSV file (optional)
            visualize: Whether to display video frames with overlays
            save_video: Whether to save annotated video
            verbose: Whether to show per-frame stats and detailed output
            
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
        
        # Update object detector verbose setting
        self.object_detector.verbose = verbose
        
        if verbose:
            print(f"Processing video: {video_path}")
            print(f"Output will be saved to: {output_path}")
        
        start_time = time.time()
        
        # Process video
        with VideoProcessor(str(video_path)) as video_processor:
            video_info = video_processor.get_video_info()
            if verbose:
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
            frame_iter = video_processor.iterate_frames()
            if verbose:
                frame_iter = tqdm(frame_iter, desc="Processing frames")
            for frame_number, frame in frame_iter:
                # Calculate timestamp based on processed frame rate
                timestamp = frame_number / video_info['target_fps']
                
                # Step 1: Detect all objects using YOLO (persons, chairs, etc.)
                object_detections = self.object_detector.detect_objects(frame)
                
                # Step 2: Extract person detections and track them ONCE
                person_detections = self.person_detector.detect_persons(frame, object_detections)
                
                # Step 3: Track persons first (to get IDs) - only once
                tracked_persons = self.person_detector.track_persons(
                    frame, person_detections, timestamp, pose_detections=None
                )
                
                # Step 4: Detect poses for tracked persons
                if self.use_holistic_pose:
                    # Use holistic model for pose detection
                    person_bboxes = [person['bbox'] for person in tracked_persons]
                    holistic_pose_detections = self.holistic_detector.detect_poses_for_persons(frame, person_bboxes)
                    
                    # Match holistic poses to tracked persons
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
                
                # Step 5: Update change detection with pose information
                # We tracked once above, now update with pose info for change detection
                # This avoids calling track_persons twice
                pose_map_by_id = {pose_det['person_id']: pose_det for pose_det in pose_detections}
                for person in tracked_persons:
                    person_id = person['person_id']
                    current_centroid = person.get('centroid')
                    if not current_centroid:
                        current_centroid = self.person_detector.calculate_centroid(person['bbox'])
                    
                    # Get pose for this person
                    pose_det = pose_map_by_id.get(person_id)
                    current_pose = pose_det.get('landmark_coords') if pose_det else None
                    
                    # Check for significant change and update signatures if needed
                    # This is what the second track_persons call would do
                    if self.person_detector._detect_significant_change(person_id, current_centroid, current_pose, timestamp):
                        sig_data = self.person_detector.person_signatures.get(person_id)
                        if sig_data:
                            features = person.get('features')
                            if features is not None:
                                sig_data['bboxes'].append(person['bbox'].copy())
                                sig_data['features'].append(features.copy())
                                
                                # Limit to max
                                if len(sig_data['bboxes']) > self.person_detector.max_signatures_per_person:
                                    n = self.person_detector.max_signatures_per_person
                                    sig_data['bboxes'] = sig_data['bboxes'][-n:]
                                    sig_data['features'] = sig_data['features'][-n:]
                
                # Store frame samples for tagging (store first frame for each new person)
                for person in tracked_persons:
                    person_id = person['person_id']
                    if person_id not in self.frame_samples:
                        # Extract person region from frame
                        x1, y1, x2, y2 = map(int, person['bbox'])
                        person_region = frame[y1:y2, x1:x2].copy()
                        if person_region.size > 0:
                            self.frame_samples[person_id] = person_region
                
                # Map person IDs to identifiers for CSV export
                # Update object_detections to use identifiers for persons
                person_id_to_identifier = {}
                for person in tracked_persons:
                    person_id = person.get('person_id')
                    identifier = person.get('identifier')
                    if identifier and not identifier.startswith('person_'):
                        person_id_to_identifier[person_id] = identifier
                
                # Update object_detections with identifiers for persons
                updated_object_detections = []
                for detection in object_detections:
                    if detection['class_name'] == 'person':
                        # Find matching tracked person
                        for person in tracked_persons:
                            p_x1, p_y1, p_x2, p_y2 = map(int, person['bbox'])
                            d_x1, d_y1, d_x2, d_y2 = map(int, detection['bbox'])
                            # Check if bboxes match (simple overlap check)
                            if abs(d_x1 - p_x1) < 10 and abs(d_y1 - p_y1) < 10:
                                identifier = person.get('identifier')
                                if identifier and not identifier.startswith('person_'):
                                    # Use identifier instead of "person"
                                    updated_detection = detection.copy()
                                    updated_detection['class_name'] = identifier
                                    updated_object_detections.append(updated_detection)
                                else:
                                    updated_object_detections.append(detection)
                                break
                        else:
                            # No matching tracked person found, keep original
                            updated_object_detections.append(detection)
                    else:
                        updated_object_detections.append(detection)
                
                # Add to CSV export with updated detections
                self.csv_exporter.add_frame_data(
                    frame_number, timestamp, updated_object_detections, pose_detections
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
        
        # Save all detected persons to signature file (with temporary IDs if not tagged)
        # Skip interactive tagging during normal video processing - use --only-tagging for that
        if self.enable_tagging and self.signature_manager:
            # Get unique persons
            unique_persons = self.person_detector.get_unique_persons_for_tagging()
            
            if unique_persons:
                if verbose:
                    print(f"\nFound {len(unique_persons)} unique person(s) in video.")
                    print("Saving signatures with temporary IDs. Use --only-tagging to tag them later.\n")
                else:
                    print(f"\nFound {len(unique_persons)} unique person(s). Saving signatures...")
                
                # Save all persons with temporary IDs (no interactive tagging during processing)
                for person_data in unique_persons:
                    person_id = person_data['person_id']
                    
                    # Get sample image for this person
                    sample_image = self.frame_samples.get(person_id)
                    if verbose:
                        if sample_image is not None:
                            print(f"Found sample image for person {person_id}: shape {sample_image.shape}")
                        else:
                            print(f"WARNING: No sample image found for person {person_id}")
                    
                    # Get signature data
                    sig_data = self.person_detector.person_signatures.get(person_id)
                    if not sig_data:
                        continue
                    
                    # Use temporary identifier (will be tagged later with --only-tagging)
                    temp_identifier = f"person_{person_id}"
                    
                    # Save to signature manager
                    bboxes = sig_data.get('bboxes', [])
                    features = sig_data.get('features', [])
                    
                    if bboxes and features:
                        # Save all signatures for this person
                        for i, (bbox, feature) in enumerate(zip(bboxes, features)):
                            sample = sample_image if i == 0 else None
                            if sample is not None and sample.size > 0:
                                self.signature_manager.add_signature(
                                    temp_identifier, bbox, feature, sample
                                )
                            else:
                                # Still save signature even without image
                                self.signature_manager.add_signature(
                                    temp_identifier, bbox, feature, None
                                )
                        
                        # Update identifier in person_detector
                        sig_data['identifier'] = temp_identifier
                        if verbose:
                            if sample_image is not None:
                                print(f"Saved person {person_id} with temporary ID: {temp_identifier} (with image)")
                            else:
                                print(f"Saved person {person_id} with temporary ID: {temp_identifier} (no image)")
                    
                # Save all signatures to file
                self.signature_manager.save_signatures()
                print(f"\nSaved {len(unique_persons)} person signature(s) to {self.signature_manager.signature_file}")
                if verbose:
                    print("Run with --only-tagging to tag these persons with identifiers.")
            else:
                if verbose:
                    print("\nNo unique persons found in video.")
            
            # Also save any persons that were detected but not in unique_persons list
            # (e.g., if they were already matched to known persons)
            if self.signature_manager:
                # Save any remaining persons from person_detector that weren't saved yet
                for person_id, sig_data in self.person_detector.person_signatures.items():
                    if sig_data.get('bboxes') and sig_data.get('features'):
                        identifier = sig_data.get('identifier')
                        if not identifier:
                            # Save with temporary identifier
                            temp_identifier = f"person_{person_id}"
                            sample_image = self.frame_samples.get(person_id)
                            bboxes = sig_data.get('bboxes', [])
                            features = sig_data.get('features', [])
                            
                            for i, (bbox, feature) in enumerate(zip(bboxes, features)):
                                sample = sample_image if i == 0 else None
                                self.signature_manager.add_signature(
                                    temp_identifier, bbox, feature, sample
                                )
                            
                            sig_data['identifier'] = temp_identifier
                
                # Save all signatures
                self.signature_manager.save_signatures()
        
        # Get processing statistics
        processing_time = time.time() - start_time
        stats = self.csv_exporter.get_summary_stats()
        stats.update({
            'processing_time': processing_time,
            'fps_processed': frame_count / processing_time if processing_time > 0 else 0,
            'video_info': video_info
        })
        
        if verbose:
            print(f"\nProcessing completed in {processing_time:.2f} seconds")
            print(f"Processed {frame_count} frames at {stats['fps_processed']:.2f} FPS")
            print(f"Summary: {stats}")
        else:
            # Minimal output for non-verbose mode
            print(f"Processing completed: {frame_count} frames in {processing_time:.2f}s ({stats['fps_processed']:.2f} FPS)")
        
        return stats
        
    def process_folder(self, folder_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Process all videos in a folder.
        
        Args:
            folder_path: Path to folder containing videos
            verbose: Whether to show per-frame stats and detailed output
            
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
            if verbose:
                print(f"No video files found in {folder_path}")
            return []
            
        if verbose:
            print(f"Found {len(video_files)} video files to process")
        else:
            print(f"Processing {len(video_files)} video files...")
        
        results = []
        for video_file in video_files:
            try:
                if verbose:
                    print(f"\nProcessing: {video_file.name}")
                result = self.process_video(str(video_file), verbose=verbose)
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
        
        # Create mapping from person bbox to identifier
        person_bbox_to_identifier = {}
        for person in tracked_persons:
            person_id = person.get('person_id')
            identifier = person.get('identifier')
            if identifier and not identifier.startswith('person_'):
                # Use bbox as key for matching
                bbox_key = tuple(map(int, person['bbox']))
                person_bbox_to_identifier[bbox_key] = identifier
        
        # Draw object detections
        for detection in object_detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # For persons, check if we have an identifier
            display_name = class_name
            if class_name == 'person':
                # Check if this detection matches a tracked person with identifier
                bbox_key = (x1, y1, x2, y2)
                # Try to find matching bbox (with small tolerance)
                identifier = None
                for p_bbox, p_identifier in person_bbox_to_identifier.items():
                    p_x1, p_y1, p_x2, p_y2 = p_bbox
                    if abs(x1 - p_x1) < 10 and abs(y1 - p_y1) < 10:
                        identifier = p_identifier
                        break
                
                if identifier:
                    display_name = identifier
                    color = (0, 255, 0)  # Green for identified person
                else:
                    color = (0, 255, 0)  # Green for person
            elif class_name == 'chair':
                color = (255, 0, 0)  # Blue for chair
            else:
                color = (0, 0, 255)  # Red for other objects
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{display_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw person identifiers on tracking boxes
        for person in tracked_persons:
            x1, y1, x2, y2 = map(int, person['bbox'])
            person_id = person.get('person_id')
            identifier = person.get('identifier')
            
            # Display identifier if available, otherwise person_id
            if identifier and not identifier.startswith('person_'):
                display_text = identifier
            else:
                display_text = f"Person {person_id}"
            
            # Draw identifier below bounding box (in yellow)
            cv2.putText(vis_frame, display_text, (x1, y2 + 20), 
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
        
    def tag_persons_only(self) -> Dict[str, Any]:
        """
        Tag persons from stored signatures without processing video.
        This mode loads signatures and presents them for tagging.
        
        Returns:
            Dictionary with tagging results
        """
        if not self.signature_manager or not self.tagging_ui:
            raise ValueError("Tagging is not enabled. Initialize pipeline with enable_tagging=True")
        
        # Load signatures if not already loaded
        self.signature_manager.load_signatures()
        
        # Get all untagged persons
        untagged_persons = self.signature_manager.get_all_untagged_for_tagging()
        
        if not untagged_persons:
            print("\nNo untagged persons found in signature file.")
            return {'tagged_count': 0, 'total_persons': len(self.signature_manager.known_persons)}
        
        print(f"\nFound {len(untagged_persons)} untagged person(s) in signature file.")
        print("Please tag each person with an identifier (name/ID).\n")
        
        # Prepare person data for UI (convert to format expected by UI)
        person_data_list = []
        frame_samples = {}
        identifier_map = {}  # Map display person_id to old_identifier
        
        for idx, person_data in enumerate(untagged_persons):
            # Use index as person_id for display purposes
            person_id = idx
            old_identifier = person_data['identifier']
            identifier_map[person_id] = old_identifier
            
            # Get first sample image if available
            sample_images = person_data.get('sample_images', [])
            if sample_images and len(sample_images) > 0:
                # Convert numpy array if needed (pickle might return it as list)
                sample_img = sample_images[0]
                if isinstance(sample_img, list):
                    sample_img = np.array(sample_img, dtype=np.uint8)
                # Ensure it's a valid numpy array
                if isinstance(sample_img, np.ndarray) and sample_img.size > 0:
                    frame_samples[person_id] = sample_img
                    print(f"Loaded sample image for {old_identifier}: shape {sample_img.shape}")
                else:
                    print(f"WARNING: Invalid sample image for {old_identifier}")
            else:
                print(f"WARNING: No sample images found for {old_identifier} (sample_images={sample_images})")
            
            person_data_list.append({
                'person_id': person_id,  # Temporary ID for display
                'identifier': old_identifier,  # Current identifier
                'bboxes': person_data.get('bboxes', []),
                'features': person_data.get('features', []),
                'num_signatures': len(person_data.get('bboxes', []))
            })
        
        # Tag persons using UI (allow_temporary_ids=True since we want to tag persons with temp IDs like "person_0")
        person_identifiers = self.tagging_ui.tag_persons(person_data_list, frame_samples, allow_temporary_ids=True)
        
        # Update identifiers in signature manager
        tagged_count = 0
        for person_id, new_identifier in person_identifiers.items():
            old_identifier = identifier_map.get(person_id)
            if new_identifier and new_identifier != old_identifier:
                # Update identifier (replace temporary ID with real identifier)
                success = self.signature_manager.update_identifier(old_identifier, new_identifier)
                if success:
                    print(f"Updated identifier: '{old_identifier}' -> '{new_identifier}'")
                    tagged_count += 1
            # If new_identifier is None/empty, person was skipped - keep the temporary ID
        
        # Save updated signatures
        if tagged_count > 0:
            self.signature_manager.save_signatures()
            print(f"\nSaved {tagged_count} updated signature(s) to {self.signature_manager.signature_file}")
        else:
            print("\nNo identifiers were updated.")
        
        return {
            'tagged_count': tagged_count,
            'total_persons': len(self.signature_manager.known_persons),
            'untagged_count': len(untagged_persons)
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.use_holistic_pose:
            if hasattr(self.holistic_detector, 'holistic'):
                self.holistic_detector.holistic.close()
        else:
            if hasattr(self.pose_detector, 'pose'):
                self.pose_detector.pose.close()
        self.csv_exporter.clear_data()
