#!/usr/bin/env python3
"""
Profile PersonDetector.detect_persons() to identify performance bottlenecks.
"""

import cProfile
import pstats
import io
import numpy as np
import time
from typing import List, Dict


# Original implementation (simplified for profiling)
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
        
        # Sort by confidence and limit to max_persons
        person_detections.sort(key=lambda x: x['confidence'], reverse=True)
        return person_detections[:self.max_persons]


# Optimized version 1: List comprehension + nlargest
class PersonDetectorOptimizedV1:
    def __init__(self, max_persons=3):
        self.max_persons = max_persons
    
    def detect_persons(self, frame: np.ndarray, object_detections: List[Dict]) -> List[Dict]:
        """Optimized V1: List comprehension + nlargest."""
        from heapq import nlargest
        
        # Filter and create person detections in one pass
        person_detections = [
            {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_name': 'person',
                'person_id': None,
                'features': None
            }
            for det in object_detections
            if det.get('class_name') == 'person'
        ]
        
        # Use nlargest for partial sort when limiting
        if len(person_detections) > self.max_persons:
            person_detections = nlargest(
                self.max_persons,
                person_detections,
                key=lambda x: x['confidence']
            )
            person_detections.reverse()
        else:
            person_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return person_detections


# Optimized version 2: Minimal dict creation, reuse original dict
class PersonDetectorOptimizedV2:
    def __init__(self, max_persons=3):
        self.max_persons = max_persons
    
    def detect_persons(self, frame: np.ndarray, object_detections: List[Dict]) -> List[Dict]:
        """Optimized V2: Reuse original dict, minimal copying."""
        if not object_detections:
            return []
        
        # Filter persons in one pass, reuse original dicts
        persons = []
        persons_append = persons.append  # Cache method lookup
        
        for det in object_detections:
            if det.get('class_name') == 'person':
                persons_append(det)
        
        if not persons:
            return []
        
        # Sort by confidence
        persons.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit results
        result = persons[:self.max_persons]
        
        # Add missing fields only to returned items (in-place modification)
        for p in result:
            if 'person_id' not in p:
                p['person_id'] = None
            if 'features' not in p:
                p['features'] = None
            if 'class_name' not in p:
                p['class_name'] = 'person'
        
        return result


# Optimized version 3: Early exit, tuple-based sorting
class PersonDetectorOptimizedV3:
    def __init__(self, max_persons=3):
        self.max_persons = max_persons
    
    def detect_persons(self, frame: np.ndarray, object_detections: List[Dict]) -> List[Dict]:
        """Optimized V3: Tuple-based sorting, minimal allocations."""
        if not object_detections:
            return []
        
        # Pre-allocate list
        person_tuples = []
        person_tuples_append = person_tuples.append
        
        # Single pass: filter and create tuples (confidence first for sorting)
        for det in object_detections:
            if det.get('class_name') == 'person':
                person_tuples_append((
                    det['confidence'],  # Sort key first
                    det['bbox'],
                    det  # Keep reference to original dict
                ))
        
        if not person_tuples:
            return []
        
        # Sort by confidence (first element) - descending
        person_tuples.sort(reverse=True)
        
        # Limit and convert to dict format
        result = []
        result_append = result.append
        for conf, bbox, det in person_tuples[:self.max_persons]:
            result_append({
                'bbox': bbox,
                'confidence': conf,
                'class_name': 'person',
                'person_id': None,
                'features': None
            })
        
        return result


def generate_mock_object_detections(num_persons=5, num_other_objects=10):
    """Generate mock object detections for testing."""
    detections = []
    
    # Add person detections
    for i in range(num_persons):
        detections.append({
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.5 + (i * 0.1),
            'bbox': [100.0 + i*50, 100.0 + i*30, 200.0 + i*50, 300.0 + i*30]
        })
    
    # Add other object detections
    other_classes = ['chair', 'bottle', 'cup', 'laptop', 'book']
    for i in range(num_other_objects):
        detections.append({
            'class_id': i % 5 + 1,
            'class_name': other_classes[i % len(other_classes)],
            'confidence': 0.4 + (i * 0.05),
            'bbox': [50.0 + i*30, 50.0 + i*20, 150.0 + i*30, 200.0 + i*20]
        })
    
    return detections


def profile_function(func, detector, frame, object_detections, num_iterations=10000, name="Function"):
    """Profile a function."""
    print(f"\n{'='*60}")
    print(f"Profiling {name} - {num_iterations} iterations")
    print(f"{'='*60}\n")
    
    # Warm-up run
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
    
    # Print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(15)  # Top 15 functions
    print(s.getvalue())
    
    return result, avg_time


def call_detect_persons(detector, frame, object_detections):
    """Wrapper to call detect_persons."""
    return detector.detect_persons(frame, object_detections)


def main():
    """Main profiling function."""
    print("="*60)
    print("PersonDetector.detect_persons() Performance Profiling")
    print("="*60)
    
    # Setup
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    num_iterations = 100000  # High iteration count for accurate profiling
    
    # Test with different numbers of detections
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
        
        # Create detector instances
        original = PersonDetectorOriginal(max_persons=3)
        optimized_v1 = PersonDetectorOptimizedV1(max_persons=3)
        optimized_v2 = PersonDetectorOptimizedV2(max_persons=3)
        optimized_v3 = PersonDetectorOptimizedV3(max_persons=3)
        
        # Profile original
        original_result, original_time = profile_function(
            call_detect_persons, original, frame, object_detections,
            num_iterations, "ORIGINAL"
        )
        
        # Profile optimized versions
        opt_v1_result, opt_v1_time = profile_function(
            call_detect_persons, optimized_v1, frame, object_detections,
            num_iterations, "OPTIMIZED V1 (list comprehension + nlargest)"
        )
        
        opt_v2_result, opt_v2_time = profile_function(
            call_detect_persons, optimized_v2, frame, object_detections,
            num_iterations, "OPTIMIZED V2 (reuse dict + in-place)"
        )
        
        opt_v3_result, opt_v3_time = profile_function(
            call_detect_persons, optimized_v3, frame, object_detections,
            num_iterations, "OPTIMIZED V3 (tuple sorting)"
        )
        
        # Compare results
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"Original:        {original_time*1000:.3f} ms/call")
        print(f"Optimized V1:    {opt_v1_time*1000:.3f} ms/call ({original_time/opt_v1_time:.2f}x faster)")
        print(f"Optimized V2:    {opt_v2_time*1000:.3f} ms/call ({original_time/opt_v2_time:.2f}x faster)")
        print(f"Optimized V3:    {opt_v3_time*1000:.3f} ms/call ({original_time/opt_v3_time:.2f}x faster)")
        
        # Verify results are equivalent
        print(f"\n{'='*60}")
        print("RESULT VERIFICATION")
        print(f"{'='*60}")
        print(f"Original result count: {len(original_result)}")
        print(f"Optimized V1 count:   {len(opt_v1_result)}")
        print(f"Optimized V2 count:   {len(opt_v2_result)}")
        print(f"Optimized V3 count:   {len(opt_v3_result)}")
        
        # Check if results match
        if len(original_result) == len(opt_v2_result) == len(opt_v3_result):
            print("✓ Result counts match")
        else:
            print("✗ Result counts differ!")


if __name__ == "__main__":
    main()
