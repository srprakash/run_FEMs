#!/usr/bin/env python3
"""
Compare original vs optimized ObjectDetector.detect_objects() performance.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import both versions
from models.object_detector import ObjectDetector


def generate_test_frame(width=640, height=480):
    """Generate a test frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def benchmark_detect_objects(detector, frame, num_iterations=50, name="Detector"):
    """Benchmark detect_objects method."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {name}")
    print(f"{'='*60}")
    
    # Warm-up
    _ = detector.detect_objects(frame)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = detector.detect_objects(frame)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"Average time: {avg_time*1000:.3f} ms (±{std_time*1000:.3f} ms)")
    print(f"Min time: {min_time*1000:.3f} ms")
    print(f"Max time: {max_time*1000:.3f} ms")
    print(f"Throughput: {1.0/avg_time:.1f} FPS")
    print(f"Detections: {len(result)}")
    
    return avg_time, result


def main():
    """Compare original and optimized versions."""
    print("="*60)
    print("ObjectDetector.detect_objects() Performance Comparison")
    print("="*60)
    
    try:
        detector = ObjectDetector(verbose=False)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Test with different frame sizes
    test_cases = [
        ("Small (320x240)", 320, 240),
        ("Medium (640x480)", 640, 480),
        ("Large (1280x720)", 1280, 720),
    ]
    
    num_iterations = 30
    
    for test_name, width, height in test_cases:
        print(f"\n{'#'*60}")
        print(f"Test: {test_name}")
        print(f"{'#'*60}")
        
        frame = generate_test_frame(width, height)
        
        # Benchmark optimized version (current implementation)
        opt_time, opt_result = benchmark_detect_objects(
            detector, frame, num_iterations, "OPTIMIZED (Current)"
        )
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Frame size: {width}x{height}")
        print(f"Average time: {opt_time*1000:.3f} ms")
        print(f"FPS: {1.0/opt_time:.1f}")
        print(f"Detections: {len(opt_result)}")


if __name__ == "__main__":
    main()

