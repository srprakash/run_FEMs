#!/usr/bin/env python3
"""
Main entry point for the video processing pipeline.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from pipeline import VideoProcessingPipeline


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Video Processing Pipeline - Analyze videos with computer vision models"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=False,
        help="Input video file or folder path (not required for --only-tagging mode)"
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
        "--only-tagging",
        action="store_true",
        help="Tag persons from stored signatures without processing video. Requires --output-dir to locate signature file."
    )
    
    parser.add_argument(
        "--signature-file",
        default="person_signatures.pkl",
        help="Name of signature file (default: person_signatures.pkl)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle only-tagging mode
    if args.only_tagging:
        try:
            pipeline = VideoProcessingPipeline(
                output_dir=args.output_dir,
                signature_file=args.signature_file,
                enable_tagging=True
            )
            
            print("Tagging-only mode: Loading signatures and presenting for tagging...")
            result = pipeline.tag_persons_only()
            
            print(f"\nTagging completed!")
            print(f"  Tagged persons: {result.get('tagged_count', 0)}")
            print(f"  Total persons: {result.get('total_persons', 0)}")
            print(f"  Untagged persons: {result.get('untagged_count', 0)}")
            
            pipeline.cleanup()
            return
            
        except Exception as e:
            print(f"Error in tagging mode: {e}")
            sys.exit(1)
    
    # Validate input path for normal processing
    if not args.input:
        print("Error: --input is required when not using --only-tagging mode")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = VideoProcessingPipeline(
            output_dir=args.output_dir,
            signature_file=args.signature_file,
            verbose=args.verbose
        )
        
        if args.verbose:
            model_info = pipeline.get_model_info()
            print("Model Information:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            print()
        
        if args.batch or input_path.is_dir():
            # Batch processing
            if args.verbose:
                print(f"Processing all videos in: {input_path}")
            results = pipeline.process_folder(str(input_path), verbose=args.verbose)
            
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
            if args.verbose:
                print(f"Processing single video: {input_path}")
            if args.visualize:
                print("Press 'q' to quit visualization")
            result = pipeline.process_video(
                str(input_path), 
                args.output, 
                visualize=args.visualize,
                save_video=args.save_video,
                verbose=args.verbose
            )
            
            if args.verbose:
                print(f"\nProcessing completed successfully!")
                print(f"  Processing time: {result['processing_time']:.2f} seconds")
                print(f"  FPS processed: {result['fps_processed']:.2f}")
                print(f"  Total frames: {result['video_info']['frame_count']}")
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
