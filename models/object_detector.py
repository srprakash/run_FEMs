"""
Object detection module using YOLO or ONNX Runtime.
Automatically uses ONNX Runtime on non-GPU platforms (Intel Mac, Raspberry Pi).
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from config import MAX_OBJECTS_PER_FRAME, CONFIDENCE_THRESHOLD

# Platform detection
try:
    from utils.platform_detection import is_non_gpu_platform
except ImportError:
    # Fallback if platform detection not available
    def is_non_gpu_platform():
        return False

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try to import YOLO (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector:
    """
    Object detection using YOLO or ONNX Runtime.
    Automatically selects ONNX Runtime on non-GPU platforms (Intel Mac, Raspberry Pi).
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", verbose: bool = False, 
                 force_onnx: bool = None, onnx_model_path: str = None):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file (.pt) - used if ONNX not available/forced
            verbose: Whether to show verbose output
            force_onnx: Force use of ONNX Runtime (True) or YOLO (False). 
                       If None, auto-detects based on platform.
            onnx_model_path: Path to ONNX model file (.onnx). If None, tries to find
                           model_path with .onnx extension.
        """
        self.max_objects = MAX_OBJECTS_PER_FRAME
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.verbose = verbose
        self.use_onnx = False
        self.model = None
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_output_names = None
        self.onnx_input_shape = None
        self.onnx_input_height = 640
        self.onnx_input_width = 640
        self.class_names = {}
        
        # Determine which backend to use
        if force_onnx is None:
            # Auto-detect: use ONNX on non-GPU platforms if available
            use_onnx = is_non_gpu_platform() and ONNX_AVAILABLE
        else:
            use_onnx = force_onnx
        
        if use_onnx and ONNX_AVAILABLE:
            # Use ONNX Runtime
            self._init_onnx(model_path, onnx_model_path)
        elif YOLO_AVAILABLE:
            # Use YOLO (ultralytics)
            self._init_yolo(model_path)
        else:
            raise ImportError(
                "Neither ONNX Runtime nor YOLO (ultralytics) is available. "
                "Install one of: pip install onnxruntime or pip install ultralytics"
            )
    
    def _init_onnx(self, model_path: str, onnx_model_path: str = None):
        """Initialize ONNX Runtime backend."""
        self.use_onnx = True
        
        # Find ONNX model path
        if onnx_model_path:
            onnx_path = Path(onnx_model_path)
        else:
            # Try to find .onnx version of model_path
            pt_path = Path(model_path)
            onnx_path = pt_path.with_suffix('.onnx')
            
            # If .onnx doesn't exist, try common locations
            if not onnx_path.exists():
                # Try in same directory
                alt_path = pt_path.parent / "yolov8n.onnx"
                if alt_path.exists():
                    onnx_path = alt_path
                else:
                    # Try current directory
                    alt_path = Path("yolov8n.onnx")
                    if alt_path.exists():
                        onnx_path = alt_path
        
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}. "
                f"Please convert YOLO model to ONNX format or provide onnx_model_path."
            )
        
        if self.verbose:
            print(f"Using ONNX Runtime with model: {onnx_path}")
        
        # Setup execution providers - use CPU only for non-GPU platforms
        providers = ['CPUExecutionProvider']
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        if not self.verbose:
            sess_options.log_severity_level = 3  # Suppress warnings
        
        # Enable optimizations
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.onnx_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output details
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_input_shape = self.onnx_session.get_inputs()[0].shape
        self.onnx_output_names = [output.name for output in self.onnx_session.get_outputs()]
        
        # Get model input size
        if len(self.onnx_input_shape) == 4:
            _, _, self.onnx_input_height, self.onnx_input_width = self.onnx_input_shape
        else:
            self.onnx_input_height = self.onnx_input_width = 640
        
        # Load class names (COCO dataset classes for YOLO)
        self.class_names = self._load_coco_class_names()
        
        if self.verbose:
            actual_provider = self.onnx_session.get_providers()[0]
            print(f"ONNX Runtime initialized with provider: {actual_provider}")
            print(f"Model input: {self.onnx_input_name}, shape: {self.onnx_input_shape}")
            print(f"Model outputs: {self.onnx_output_names}")
    
    def _init_yolo(self, model_path: str):
        """Initialize YOLO backend."""
        self.use_onnx = False
        
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO (ultralytics) is not available. Install with: pip install ultralytics")
        
        if self.verbose:
            print(f"Using YOLO (ultralytics) with model: {model_path}")
        
        self.model = YOLO(model_path)
        self.class_names = self.model.names
    
    def _load_coco_class_names(self) -> Dict[int, str]:
        """Load COCO class names (YOLO uses COCO dataset)."""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        return {i: name for i, name in enumerate(coco_classes)}
    
    def _preprocess_frame_onnx(self, frame: np.ndarray) -> tuple:
        """
        Preprocess frame for YOLO ONNX inference.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            (preprocessed_tensor, original_shape, scale_factor)
        """
        original_h, original_w = frame.shape[:2]
        
        # Resize to model input size (maintain aspect ratio)
        scale = min(self.onnx_input_width / original_w, self.onnx_input_height / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to model input size (center padding)
        padded = np.full((self.onnx_input_height, self.onnx_input_width, 3), 114, dtype=np.uint8)  # Gray padding
        pad_h = (self.onnx_input_height - new_h) // 2
        pad_w = (self.onnx_input_width - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        chw = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension: (1, 3, H, W)
        batched = np.expand_dims(chw, axis=0)
        
        return batched, (original_h, original_w), (scale, pad_w, pad_h)
    
    def _postprocess_onnx_outputs(self, outputs: List[np.ndarray], 
                                  original_shape: tuple, 
                                  scale_info: tuple) -> List[Dict]:
        """
        Postprocess YOLO ONNX outputs to extract detections.
        
        Args:
            outputs: Model outputs (usually [boxes, scores, class_ids] or single output)
            original_shape: (height, width) of original frame
            scale_info: (scale, pad_w, pad_h) from preprocessing
            
        Returns:
            List of detection dictionaries
        """
        scale, pad_w, pad_h = scale_info
        original_h, original_w = original_shape
        
        # Debug: Check outputs (only in verbose mode)
        if self.verbose:
            print(f"[ONNX] Number of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"[ONNX] Output {i} shape: {out.shape}, dtype: {out.dtype}")
        
        # YOLO ONNX output format varies by version:
        # - YOLOv8: Single output (1, N, 84) where 84 = 4 (bbox) + 80 (classes)
        #           Format: [x_center, y_center, width, height, class_0_prob, ..., class_79_prob]
        # - Older YOLO: Multiple outputs (boxes, scores, class_ids)
        
        boxes = None
        class_ids = None
        confidences = None
        
        # Handle different output formats
        if len(outputs) == 1:
            # YOLOv8 format: single output
            predictions = outputs[0]
            
            # Handle different shape formats:
            # - (1, 8400, 84) - standard format: batch, detections, features
            # - (1, 84, 8400) - transposed format: batch, features, detections
            # - (8400, 84) - no batch dimension
            if len(predictions.shape) == 3:
                batch_size, dim1, dim2 = predictions.shape
                if dim1 == 84 and dim2 == 8400:
                    # Transposed format: (1, 84, 8400) -> transpose to (1, 8400, 84)
                    predictions = predictions.transpose(0, 2, 1)  # (1, 8400, 84)
                    predictions = predictions[0]  # (8400, 84)
                elif dim1 == 8400 and dim2 == 84:
                    # Standard format: (1, 8400, 84)
                    predictions = predictions[0]  # (8400, 84)
                else:
                    if self.verbose:
                        print(f"[ONNX] Unexpected 3D shape: {predictions.shape}, trying to handle...")
                    # Try to infer: if first dim is 1, remove it and check remaining
                    if batch_size == 1:
                        predictions = predictions[0]  # Remove batch
                        if len(predictions.shape) == 2:
                            # Check if we need to transpose
                            if predictions.shape[0] < predictions.shape[1]:
                                predictions = predictions.T
            elif len(predictions.shape) == 2:
                # Already 2D: check if we need to transpose
                if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] == 84:
                    # Likely transposed: (84, 8400) -> transpose to (8400, 84)
                    predictions = predictions.T
            else:
                if self.verbose:
                    print(f"[ONNX] Unexpected predictions shape: {predictions.shape}")
                return []
            
            # Check if we have any predictions
            if predictions.shape[0] == 0:
                if self.verbose:
                    print("[ONNX] No predictions in output")
                return []
            
            # Verify shape is now (N, 84)
            if predictions.shape[1] != 84:
                if self.verbose:
                    print(f"[ONNX] Unexpected feature dimension: {predictions.shape}, expected (N, 84)")
                return []
            
            # Extract boxes and class probabilities
            boxes = predictions[:, :4]  # (N, 4) - x_center, y_center, width, height
            class_probs = predictions[:, 4:]  # (N, 80) - class probabilities
            
            # Get class IDs and confidences
            class_ids = np.argmax(class_probs, axis=1)  # (N,)
            confidences = np.max(class_probs, axis=1)  # (N,) - class confidence
            
            if self.verbose and len(confidences) > 0:
                print(f"[ONNX] Found {len(boxes)} raw detections, confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
            
        elif len(outputs) >= 2:
            # Separate outputs format (older YOLO versions)
            boxes_output = outputs[0]
            scores_output = outputs[1]
            
            # Remove batch dimension if present
            if len(boxes_output.shape) == 3:
                boxes = boxes_output[0]  # (N, 4)
            else:
                boxes = boxes_output
            
            if len(scores_output.shape) == 3:
                scores = scores_output[0]  # (N, num_classes)
            else:
                scores = scores_output
            
            # Get class IDs and confidences
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
        
        else:
            if self.verbose:
                print(f"[ONNX] Unexpected number of outputs: {len(outputs)}")
            return []
        
        # Check if we have valid detections
        if boxes is None or len(boxes) == 0:
            if self.verbose:
                print("[ONNX] No boxes found in output")
            return []
        
        # Filter by confidence threshold BEFORE processing
        valid_mask = confidences >= self.confidence_threshold
        if not np.any(valid_mask):
            if self.verbose:
                print(f"[ONNX] No detections above confidence threshold {self.confidence_threshold}")
                print(f"[ONNX] Max confidence: {confidences.max():.3f}")
            return []
        
        # Apply mask
        boxes = boxes[valid_mask]
        class_ids = class_ids[valid_mask]
        confidences = confidences[valid_mask]
        
        if self.verbose:
            print(f"[ONNX] {len(boxes)} detections after confidence filtering (threshold: {self.confidence_threshold})")
        
        # Convert center format to corner format and scale back
        detections = []
        skipped_invalid = 0
        skipped_tiny = 0
        for i in range(len(boxes)):
            # Convert from center format to corner format
            x_center, y_center, width, height = boxes[i]
            
            # Scale back to original image coordinates
            # Note: boxes are in model input coordinates (640x640), need to scale to original
            x_center = (x_center - pad_w) / scale
            y_center = (y_center - pad_h) / scale
            width = width / scale
            height = height / scale
            
            # Convert to corner coordinates
            x1 = max(0, x_center - width / 2)
            y1 = max(0, y_center - height / 2)
            x2 = min(original_w, x_center + width / 2)
            y2 = min(original_h, y_center + height / 2)
            
            # Skip invalid boxes (check with small tolerance for floating point)
            if x2 <= x1 + 1e-6 or y2 <= y1 + 1e-6:
                skipped_invalid += 1
                if self.verbose and skipped_invalid <= 3:  # Only show first 3
                    print(f"[ONNX] Skipping invalid box {i}: x_center={x_center:.1f}, y_center={y_center:.1f}, width={width:.1f}, height={height:.1f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                continue
            
            # Skip boxes that are too small
            if width < 5 or height < 5:
                skipped_tiny += 1
                if self.verbose and skipped_tiny <= 3:  # Only show first 3
                    print(f"[ONNX] Skipping tiny box {i}: width={width:.1f}, height={height:.1f}")
                continue
            
            detections.append({
                'class_id': int(class_ids[i]),
                'class_name': self.class_names.get(int(class_ids[i]), 'unknown'),
                'confidence': float(confidences[i]),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        if self.verbose and (skipped_invalid > 0 or skipped_tiny > 0):
            print(f"[ONNX] Skipped {skipped_invalid} invalid boxes, {skipped_tiny} tiny boxes")
        
        # Sort by confidence and limit
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        result = detections[:self.max_objects]
        
        if self.verbose:
            print(f"[ONNX] Returning {len(result)} detections")
        
        return result
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Uses ONNX Runtime on non-GPU platforms, YOLO otherwise.
        Optimizations applied for YOLO backend:
        1. Batch tensor extraction: Extract all boxes at once instead of one-by-one
        2. Vectorized operations: Process all coordinates, confidences, class IDs in batch
        3. Early limiting: Only process top-N detections before creating dicts
        4. Cached class names: Store model.names to avoid repeated lookups
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        if self.use_onnx:
            # Use ONNX Runtime backend
            preprocessed, original_shape, scale_info = self._preprocess_frame_onnx(frame)
            
            # Run inference
            outputs = self.onnx_session.run(
                self.onnx_output_names,
                {self.onnx_input_name: preprocessed}
            )
            
            # Postprocess outputs
            detections = self._postprocess_onnx_outputs(outputs, original_shape, scale_info)
            return detections
        else:
            # Use YOLO (ultralytics) backend
            # Suppress YOLO output unless verbose is enabled
            results = self.model(frame, conf=self.confidence_threshold, verbose=self.verbose)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                # OPTIMIZATION 1: Batch extract all tensors at once
                # Instead of calling .cpu().numpy() for each box individually (slow),
                # extract all boxes in a single batch operation (much faster)
                num_boxes = len(boxes)
                
                # Extract all coordinates, confidences, and class IDs in batch
                xyxy = boxes.xyxy.cpu().numpy()  # Shape: (N, 4) - all boxes at once
                confidences = boxes.conf.cpu().numpy()  # Shape: (N,) - all confidences at once
                class_ids = boxes.cls.cpu().int().cpu().numpy()  # Shape: (N,) - all class IDs at once
                
                # OPTIMIZATION 2: Sort and limit before creating dicts
                # This avoids creating dicts for detections we'll discard
                if num_boxes > self.max_objects:
                    # Use argsort to get indices of top-N by confidence (descending)
                    top_indices = np.argsort(confidences)[::-1][:self.max_objects]
                    xyxy = xyxy[top_indices]
                    confidences = confidences[top_indices]
                    class_ids = class_ids[top_indices]
                else:
                    # Still sort for consistency (even if we have fewer than max_objects)
                    sort_indices = np.argsort(confidences)[::-1]
                    xyxy = xyxy[sort_indices]
                    confidences = confidences[sort_indices]
                    class_ids = class_ids[sort_indices]
                
                # OPTIMIZATION 3: Vectorized dict creation using list comprehension
                # Cache class names lookup (self.model.names) - accessed once per class ID
                class_names = self.model.names
                detections = [
                    {
                        'class_id': int(class_id),
                        'class_name': class_names[int(class_id)],  # Use cached names dict
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confidences, class_ids)
                ]
                
                # Already sorted by confidence (descending) from argsort above
                break  # YOLO typically returns one result, so break after first
            
            return detections
        
    def get_top_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Get top confidence objects from frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of top confidence objects
        """
        detections = self.detect_objects(frame)
        return detections
        
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize object detections on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with bounding boxes drawn
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
        return vis_frame
