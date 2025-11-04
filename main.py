#!/usr/bin/env python3
"""
Main entry point for the video processing pipeline.
"""

import argparse
import sys
from pathlib import Path

from pipeline import VideoProcessingPipeline


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Video Processing Pipeline - Analyze videos with computer vision models"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file or folder path"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output CSV filename (for single video processing)"
    )
    
    parser.add_argument(
        "--output-dir", "-d",
        default="output",
        help="Output directory for processed files (default: output)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all videos in the input folder"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Display video frames with bounding boxes, labels, and pose keypoints"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video with overlays"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = VideoProcessingPipeline(output_dir=args.output_dir)
        
        if args.verbose:
            model_info = pipeline.get_model_info()
            print("Model Information:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            print()
        
        if args.batch or input_path.is_dir():
            # Batch processing
            print(f"Processing all videos in: {input_path}")
            results = pipeline.process_folder(str(input_path))
            
            # Print summary
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            
            print(f"\nBatch processing completed:")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            
            if failed > 0:
                print("\nFailed videos:")
                for result in results:
                    if not result['success']:
                        print(f"  - {result['video_file']}: {result['error']}")
                        
        else:
            # Single video processing
            print(f"Processing single video: {input_path}")
            if args.visualize:
                print("Press 'q' to quit visualization")
            result = pipeline.process_video(
                str(input_path), 
                args.output, 
                visualize=args.visualize,
                save_video=args.save_video
            )
            
            print(f"\nProcessing completed successfully!")
            print(f"  Processing time: {result['processing_time']:.2f} seconds")
            print(f"  FPS processed: {result['fps_processed']:.2f}")
            print(f"  Total frames: {result['video_info']['frame_count']}")
            
            if args.verbose:
                print(f"\nDetailed statistics:")
                for key, value in result.items():
                    if key not in ['processing_time', 'fps_processed', 'video_info']:
                        print(f"  {key}: {value}")
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
