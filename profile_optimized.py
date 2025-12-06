#!/usr/bin/env python3
"""
Profile the optimized detect_persons implementation.
"""

import cProfile
import pstats
import io
import numpy as np
import time
from typing import List, Dict


# Original implementation
class PersonDetectorOriginal:
    def __init__(self, max_persons=3):
        self.max_persons = max_persons
    
    def detect_persons(self, frame: np.ndarray, object_detections: List[Dict]) -> List[Dict]:
        """Original implementation."""
        person_detections = []
        
        for detection in object_detections:
            if detection['class_name'] == 'person':
                person_detection = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'class_name': 'person',
                    'person_id': None,
                    'features': None
                }
                person_detections.append(person_detection)
        
        person_detections.sort(key=lambda x: x['confidence'], reverse=True)
        return person_detections[:self.max_persons]


# Optimized implementation based on profiling
class PersonDetectorOptimized:
    def __init__(self, max_persons=3):
        self.max_persons = max_persons
    
    def detect_persons(self, frame: np.ndarray, object_detections: List[Dict]) -> List[Dict]:
        """Optimized version based on profiling insights."""
        # Early exit if no detections
        if not object_detections:
            return []
        
        # Use list comprehension with direct dict access (faster than .get())
        person_detections = [
            {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_name': 'person',
                'person_id': None,
                'features': None
            }
            for det in object_detections
            if det['class_name'] == 'person'  # Direct access, not .get()
        ]
        
        # Early exit if no persons found
        if not person_detections:
            return []
        
        # Sort and limit
        if len(person_detections) > self.max_persons:
            person_detections.sort(key=lambda x: x['confidence'], reverse=True)
            return person_detections[:self.max_persons]
        else:
            person_detections.sort(key=lambda x: x['confidence'], reverse=True)
            return person_detections


def generate_mock_object_detections(num_persons=5, num_other_objects=10):
    """Generate mock object detections for testing."""
    detections = []
    
    for i in range(num_persons):
        detections.append({
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.5 + (i * 0.1),
            'bbox': [100.0 + i*50, 100.0 + i*30, 200.0 + i*50, 300.0 + i*30]
        })
    
    other_classes = ['chair', 'bottle', 'cup', 'laptop', 'book']
    for i in range(num_other_objects):
        detections.append({
            'class_id': i % 5 + 1,
            'class_name': other_classes[i % len(other_classes)],
            'confidence': 0.4 + (i * 0.05),
            'bbox': [50.0 + i*30, 50.0 + i*20, 150.0 + i*30, 200.0 + i*20]
        })
    
    return detections


def profile_function(func, detector, frame, object_detections, num_iterations=100000, name="Function"):
    """Profile a function."""
    print(f"\n{'='*60}")
    print(f"Profiling {name} - {num_iterations} iterations")
    print(f"{'='*60}\n")
    
    # Warm-up
    _ = func(detector, frame, object_detections)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        result = func(detector, frame, object_detections)
    end_time = time.perf_counter()
    
    profiler.disable()
    
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / num_iterations
    
    print(f"Total time: {elapsed_time*1000:.2f} ms")
    print(f"Average time per call: {avg_time*1000:.3f} ms")
    print(f"Throughput: {num_iterations/elapsed_time:.1f} calls/sec\n")
    
    # Print top functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())
    
    return result, avg_time


def call_detect_persons(detector, frame, object_detections):
    """Wrapper to call detect_persons."""
    return detector.detect_persons(frame, object_detections)


def main():
    """Main profiling function."""
    print("="*60)
    print("Optimized PersonDetector.detect_persons() Profiling")
    print("="*60)
    
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    num_iterations = 100000
    
    test_cases = [
        ("Few detections (5 persons, 5 others)", 5, 5),
        ("Medium detections (10 persons, 20 others)", 10, 20),
        ("Many detections (20 persons, 50 others)", 20, 50),
    ]
    
    for test_name, num_persons, num_others in test_cases:
        print(f"\n{'#'*60}")
        print(f"Test Case: {test_name}")
        print(f"{'#'*60}")
        
        object_detections = generate_mock_object_detections(num_persons, num_others)
        
        original = PersonDetectorOriginal(max_persons=3)
        optimized = PersonDetectorOptimized(max_persons=3)
        
        original_result, original_time = profile_function(
            call_detect_persons, original, frame, object_detections,
            num_iterations, "ORIGINAL"
        )
        
        optimized_result, optimized_time = profile_function(
            call_detect_persons, optimized, frame, object_detections,
            num_iterations, "OPTIMIZED (list comprehension + direct access)"
        )
        
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"Original:  {original_time*1000:.3f} ms/call")
        print(f"Optimized: {optimized_time*1000:.3f} ms/call")
        if optimized_time < original_time:
            speedup = original_time / optimized_time
            print(f"Speedup:   {speedup:.2f}x faster")
        else:
            slowdown = optimized_time / original_time
            print(f"Slowdown:  {slowdown:.2f}x slower")
        
        print(f"\nResult counts: Original={len(original_result)}, Optimized={len(optimized_result)}")


if __name__ == "__main__":
    main()

