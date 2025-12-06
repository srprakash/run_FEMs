#!/usr/bin/env python3
"""
Benchmark PyTorch vs ONNX Runtime performance.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from models.object_detector import ObjectDetector
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch ObjectDetector not available")

try:
    from models.object_detector_onnx import ObjectDetectorONNX, convert_yolo_to_onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX ObjectDetector not available")


def generate_test_frame(width=640, height=480):
    """Generate a test frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def benchmark_detector(detector, frame, num_iterations=50, name="Detector"):
    """Benchmark a detector."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {name}")
    print(f"{'='*60}")
    
    # Warm-up
    _ = detector.detect_objects(frame)
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        result = detector.detect_objects(frame)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_iterations} iterations...")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p95_time = np.percentile(times, 95)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    print(f"  Min:     {min_time*1000:.2f} ms")
    print(f"  Max:     {max_time*1000:.2f} ms")
    print(f"  P95:     {p95_time*1000:.2f} ms")
    print(f"  FPS:     {1.0/avg_time:.1f}")
    print(f"  Detections: {len(result)}")
    
    return avg_time, result


def main():
    """Main benchmarking function."""
    print("="*60)
    print("PyTorch vs ONNX Runtime Performance Comparison")
    print("="*60)
    
    # Check if ONNX model exists, if not, offer to convert
    onnx_model_path = "yolov8n.onnx"
    yolo_model_path = "yolov8n.pt"
    
    if not Path(onnx_model_path).exists():
        print(f"\nONNX model not found: {onnx_model_path}")
        if Path(yolo_model_path).exists() and ONNX_AVAILABLE:
            response = input(f"Convert {yolo_model_path} to ONNX? (y/n): ")
            if response.lower() == 'y':
                print("Converting...")
                convert_yolo_to_onnx(yolo_model_path, onnx_model_path)
            else:
                print("Skipping ONNX benchmark")
                ONNX_AVAILABLE = False
        else:
            print("Skipping ONNX benchmark")
            ONNX_AVAILABLE = False
    
    # Test frame
    frame = generate_test_frame(640, 480)
    num_iterations = 50
    
    results = {}
    
    # Benchmark PyTorch
    if PYTORCH_AVAILABLE:
        try:
            pytorch_detector = ObjectDetector(yolo_model_path, verbose=False)
            pytorch_time, pytorch_result = benchmark_detector(
                pytorch_detector, frame, num_iterations, "PyTorch (Original)"
            )
            results['PyTorch'] = pytorch_time
        except Exception as e:
            print(f"Error benchmarking PyTorch: {e}")
            PYTORCH_AVAILABLE = False
    
    # Benchmark ONNX Runtime
    if ONNX_AVAILABLE:
        try:
            # Try GPU first
            try:
                onnx_detector_gpu = ObjectDetectorONNX(
                    onnx_model_path,
                    verbose=True,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                onnx_gpu_time, onnx_gpu_result = benchmark_detector(
                    onnx_detector_gpu, frame, num_iterations, "ONNX Runtime (GPU)"
                )
                results['ONNX GPU'] = onnx_gpu_time
            except Exception as e:
                print(f"GPU not available: {e}")
            
            # CPU fallback
            onnx_detector_cpu = ObjectDetectorONNX(
                onnx_model_path,
                verbose=True,
                providers=['CPUExecutionProvider']
            )
            onnx_cpu_time, onnx_cpu_result = benchmark_detector(
                onnx_detector_cpu, frame, num_iterations, "ONNX Runtime (CPU)"
            )
            results['ONNX CPU'] = onnx_cpu_time
            
        except Exception as e:
            print(f"Error benchmarking ONNX: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    if results:
        baseline = min(results.values())
        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline / time_val if time_val > 0 else 0
            print(f"{name:20s}: {time_val*1000:6.2f} ms ({speedup:.2f}x)")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if 'ONNX GPU' in results and 'PyTorch' in results:
        speedup = results['PyTorch'] / results['ONNX GPU']
        print(f"✓ ONNX Runtime (GPU) is {speedup:.2f}x faster than PyTorch")
        print("  Consider using ONNX Runtime for production")
    
    if 'ONNX CPU' in results and 'PyTorch' in results:
        speedup = results['PyTorch'] / results['ONNX CPU']
        if speedup > 1.1:
            print(f"✓ ONNX Runtime (CPU) is {speedup:.2f}x faster than PyTorch")
        else:
            print("  ONNX Runtime (CPU) provides minimal speedup")
            print("  Consider using GPU for better performance")


if __name__ == "__main__":
    main()

