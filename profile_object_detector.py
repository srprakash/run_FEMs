#!/usr/bin/env python3
"""
Profile ObjectDetector.detect_objects() to identify performance bottlenecks.
"""

import cProfile
import pstats
import io
import numpy as np
import time
import cv2
from pathlib import Path
import sys

# Add run_FEMs to path
sys.path.insert(0, str(Path(__file__).parent))

from models.object_detector import ObjectDetector
from config import MAX_OBJECTS_PER_FRAME, CONFIDENCE_THRESHOLD


def generate_test_frame(width=640, height=480):
    """Generate a test frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def profile_detect_objects_original(detector, frame, num_iterations=100):
    """Profile the original detect_objects method."""
    print(f"\n{'='*60}")
    print(f"Profiling ORIGINAL detect_objects() - {num_iterations} iterations")
    print(f"{'='*60}\n")
    
    # Warm-up run
    _ = detector.detect_objects(frame)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        result = detector.detect_objects(frame)
    end_time = time.perf_counter()
    
    profiler.disable()
    
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / num_iterations
    
    print(f"Total time: {elapsed_time*1000:.2f} ms")
    print(f"Average time per call: {avg_time*1000:.3f} ms")
    print(f"Throughput: {num_iterations/elapsed_time:.1f} calls/sec")
    print(f"Detections per call: {len(result) if result else 0}\n")
    
    # Print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    return result, avg_time


def main():
    """Main profiling function."""
    print("="*60)
    print("ObjectDetector.detect_objects() Performance Profiling")
    print("="*60)
    
    # Setup
    try:
        detector = ObjectDetector(verbose=False)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        print("Make sure YOLO model is available")
        return
    
    # Test with different frame sizes
    test_cases = [
        ("Small frame (320x240)", 320, 240),
        ("Medium frame (640x480)", 640, 480),
        ("Large frame (1280x720)", 1280, 720),
    ]
    
    num_iterations = 50  # Fewer iterations since YOLO is slower
    
    for test_name, width, height in test_cases:
        print(f"\n{'#'*60}")
        print(f"Test Case: {test_name}")
        print(f"{'#'*60}")
        
        frame = generate_test_frame(width, height)
        
        # Profile original
        original_result, original_time = profile_detect_objects_original(
            detector, frame, num_iterations
        )
        
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Average time per call: {original_time*1000:.3f} ms")
        print(f"FPS equivalent: {1.0/original_time:.1f} FPS")
        print(f"Detections found: {len(original_result)}")


if __name__ == "__main__":
    main()

