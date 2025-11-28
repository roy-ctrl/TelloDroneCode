"""
Tello EDU: Autonomous V17.0 "Adaptive Speed + Advanced Pathfinding"
===================================================================
Hardware: RTX 4060 8GB VRAM | Python 3.13

NEW FEATURES IN V17.0:
1. ADAPTIVE SPEED CONTROL: Speed automatically adjusts based on obstacle proximity
   - FAST mode when clear (>2.5m), CRUISE when cautious (1.5-2.5m), SLOW when danger (<1.5m)
2. CEILING PROTECTION: Strict altitude hold during rotation/exploration to prevent ceiling crashes
3. IMPROVED PATHFINDING: A* + String Pulling + B-Spline smoothing for smoother paths
4. LOW-LATENCY VIDEO: Optimized OpenCV capture with reduced buffer and FFMPEG settings
5. ALTITUDE LOCK DURING ROTATION: Prevents rising during 360° rotations

RETAINED FROM V16.0:
- VEERING BEHAVIOR: Smooth obstacle avoidance while maintaining forward motion
- OFFLINE MODE: No HuggingFace retry delays
- OPEN3D VISUALIZATION: Working 3D floor map
- PATH PROJECTION: Walkable path on camera view
- Floor segmentation, A* path planning
- Corner escape, thin obstacle detection
- NO BACKWARD FLIGHT in auto mode
"""

# ==========================================
# CRITICAL: Set offline mode BEFORE imports
# ==========================================
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
import time
import threading
import logging
import math
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque, Set, Dict
from enum import Enum
from queue import Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pygame
from djitellopy import Tello
from ultralytics import YOLOWorld

# Import scipy for B-spline path smoothing
try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
    print("[PATH] SciPy available for B-spline path smoothing")
except ImportError:
    SCIPY_AVAILABLE = False
    print("[PATH] SciPy not available - B-spline smoothing disabled")
    print("      Install with: pip install scipy")

# Try to import Open3D for 3D visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("[3D VIS] Open3D available for 3D floor visualization")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[3D VIS] Open3D not available - 3D visualization disabled")

# Try to import transformers for OneFormer
try:
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    ONEFORMER_AVAILABLE = True
    print("[SEG] OneFormer available for floor segmentation")
except ImportError:
    ONEFORMER_AVAILABLE = False
    print("[SEG] OneFormer not available - floor segmentation disabled")
    print("      Install with: pip install transformers")

# ==========================================
# CONFIGURATION
# ==========================================

# Depth checkpoints
DEPTH_CHECKPOINT_METRIC = 'checkpoints/depth_anything_v2_metric_hypersim_vitb.pth'
DEPTH_CHECKPOINT_RELATIVE = 'checkpoints/depth_anything_v2_vitb.pth'

# RTX 4060 Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Camera Intrinsics (960x720)
FX = 921.17
FY = 919.02
CX = 460.0
CY = 351.0

# Camera matrix for projection
CAMERA_MATRIX = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float64)

# Distortion coefficients (assuming none for Tello)
DIST_COEFFS = np.zeros(5)

# Mapping Constants
VOXEL_RES = 0.10        # 10cm per voxel
GRID_SIZE = 200         # 20x20 meters XZ plane
HEIGHT_LEVELS = 30      # 3 meters vertical
CENTER = GRID_SIZE // 2

# Floor Mapping Constants
FLOOR_VOXEL_RES = 0.05  # 5cm resolution for floor map
FLOOR_GRID_SIZE = 400   # 20x20 meters at 5cm resolution
FLOOR_CENTER = FLOOR_GRID_SIZE // 2

# ADE20K Floor Class IDs (0-indexed for model output)
FLOOR_CLASS_IDS = {
    3: "floor",           # floor, flooring
    28: "rug",            # rug, carpet, carpeting
    6: "road",            # road, route
    11: "sidewalk",       # sidewalk, pavement
    13: "ground",         # earth, ground
}

# ==========================================
# V17 OBSTACLE DISTANCE THRESHOLDS (meters)
# ==========================================
EMERGENCY_STOP_DIST = 0.5   # Emergency stop
CRITICAL_DIST = 0.8         # Critical - must turn
DANGER_DIST = 1.2           # Danger zone - slow speed
CAUTION_DIST = 1.8          # Caution zone - reduced speed  
SAFE_DIST = 2.5             # Safe zone - can cruise
CLEAR_DIST = 3.5            # All clear - can go fast

# ==========================================
# V17 ADAPTIVE SPEED SYSTEM
# ==========================================
class SpeedMode(Enum):
    STOP = "STOP"
    CREEP = "CREEP"
    SLOW = "SLOW"
    CRUISE = "CRUISE"
    FAST = "FAST"

# Speed values for each mode
SPEED_VALUES = {
    SpeedMode.STOP: 0,
    SpeedMode.CREEP: 12,
    SpeedMode.SLOW: 22,
    SpeedMode.CRUISE: 38,
    SpeedMode.FAST: 60,
}

# Yaw speeds
SPEED_YAW_SLOW = 35
SPEED_YAW_NORMAL = 50
SPEED_YAW_FAST = 70
SPEED_ESCAPE_YAW = 75

# Vertical speeds
SPEED_VERTICAL = 30
SPEED_VERTICAL_SLOW = 15

# Veering speeds (lateral movement while going forward)
SPEED_VEER_LATERAL = 25

# Approach speed
SPEED_APPROACH = 18

# ==========================================
# V17 STRICT CEILING PROTECTION
# ==========================================
CEILING_MIN_DIST = 0.4          # Minimum distance to ceiling (meters)
CEILING_CAUTION_DIST = 0.7      # Start slowing vertical when closer
MAX_ALTITUDE = 1.8              # Maximum allowed altitude (meters) - stricter
MIN_ALTITUDE = 0.3              # Minimum altitude to maintain
ALTITUDE_HOLD_TOLERANCE = 0.15  # Tolerance for altitude hold during rotation (meters)
ROTATION_ALTITUDE_LOCK = True   # Lock altitude during rotation

# Corner escape thresholds
CORNER_LEFT_THRESHOLD = 1.5
CORNER_RIGHT_THRESHOLD = 1.5
CORNER_FRONT_THRESHOLD = 1.2
CORNER_ROTATION_THRESHOLD = 2.0

# Thin obstacle detection
THIN_OBSTACLE_COLUMNS = 12
THIN_OBSTACLE_THRESHOLD = 0.8
EDGE_DETECTION_THRESHOLD = 50

# Safety margins
SAFETY_MARGIN = 1.3
TEMPORAL_SMOOTH_FRAMES = 3

# Goal approach
GOAL_CENTER_TOLERANCE = 0.10
GOAL_SIZE_TARGET = 0.20
GOAL_VERTICAL_TOLERANCE = 0.12

# Visual Odometry
VO_DEADZONE = 2.0
VO_MAX_FEATURES = 200
VO_QUALITY = 0.15
VO_MIN_DISTANCE = 5
VO_IMU_WEIGHT = 0.3

# Stuck detection
STUCK_THRESH = 0.15
STUCK_TIME = 3.0
CORNER_STUCK_TIME = 2.0
CORNER_ESCAPE_TIMEOUT = 4.0  # V18: Max seconds to spend in corner escape before forcing exit

# Segmentation settings
SEG_INFERENCE_INTERVAL = 5  # Run segmentation every N frames
SEG_INPUT_SIZE = 512        # Resize input for faster inference

# Path planning
PATH_UPDATE_INTERVAL = 0.5  # Faster path updates (was 1.0)
DEFAULT_TARGET_DISTANCE = 5.0  # Default target 5m ahead

# ==========================================
# V17 PATH SMOOTHING PARAMETERS
# ==========================================
PATH_STRING_PULL_ENABLED = True   # Enable string pulling optimization
PATH_BSPLINE_ENABLED = True       # Enable B-spline smoothing
PATH_BSPLINE_SAMPLES = 50         # Number of samples for B-spline
PATH_BSPLINE_SMOOTHNESS = 0.0     # 0 = interpolate through points

# Path visualization
PATH_LINE_COLOR = (0, 255, 255)  # Yellow
PATH_POINT_COLOR = (0, 255, 0)   # Green
PATH_START_COLOR = (255, 0, 0)   # Blue
PATH_GOAL_COLOR = (0, 0, 255)    # Red
PATH_LINE_THICKNESS = 3
PATH_POINT_RADIUS = 8

# Vertical stability (reduce bobbing)
VERTICAL_DEADZONE = 0.10  # Ignore small vertical errors
VERTICAL_DAMPING = 0.5    # Reduce vertical oscillation

# ==========================================
# V17 LOW-LATENCY VIDEO SETTINGS
# ==========================================
VIDEO_BUFFER_SIZE = 1           # Minimum buffer
VIDEO_FPS = 30                  # Target FPS
VIDEO_GRAB_TIMEOUT_MS = 50      # Timeout for frame grab
VIDEO_FFLAGS = "nobuffer"       # FFMPEG flags for low latency
VIDEO_PROBESIZE = "32"          # Reduce probe size for faster start
VIDEO_ANALYZEDURATION = "0"     # Don't analyze duration

# Logging
logging.getLogger('djitellopy').setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Set FFMPEG options for low latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|framedrop;1|strict;experimental"


# ==========================================
# ESCAPE STATE MACHINE
# ==========================================
class EscapeState(Enum):
    NONE = "none"
    CORNER_DETECTED = "corner_detected"
    ROTATING_LEFT = "rotating_left"
    ROTATING_RIGHT = "rotating_right"
    ESCAPE_COMPLETE = "escape_complete"


# ==========================================
# THREADED CAMERA (Simple, Low-Latency)
# ==========================================
class ThreadedCam:
    """Thread-safe camera capture - simple and low-latency."""
    
    def __init__(self, src: str = "udp://@0.0.0.0:11111"):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.lock = threading.Lock()

    def start(self) -> 'ThreadedCam':
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.grabbed, self.frame = grabbed, frame
                    self.frame_count += 1
                    
                    # Update FPS every second
                    elapsed = time.time() - self.fps_start_time
                    if elapsed > 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.fps_start_time = time.time()
            else:
                time.sleep(0.01)


    def get_fps(self) -> float:
        with self.lock:
            return self.fps
    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()


# ==========================================
# DEPTH MODEL LOADER
# ==========================================
def load_depth_model():
    """Load Depth Anything V2 model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    encoder = 'vitb'
    config = model_configs[encoder].copy()
    
    use_metric = os.path.exists(DEPTH_CHECKPOINT_METRIC)
    
    if use_metric:
        checkpoint_path = DEPTH_CHECKPOINT_METRIC
        print(f"[DEPTH] Found METRIC model: {checkpoint_path}")
        
        try:
            metric_depth_path = os.path.join(os.getcwd(), 'metric_depth')
            if os.path.exists(metric_depth_path) and metric_depth_path not in sys.path:
                sys.path.insert(0, metric_depth_path)
            
            from depth_anything_v2.dpt import DepthAnythingV2
            
            import inspect
            sig = inspect.signature(DepthAnythingV2.__init__)
            supports_max_depth = 'max_depth' in sig.parameters
            
            if supports_max_depth:
                config['max_depth'] = 20
                print("[DEPTH] Using metric model with max_depth=20")
            else:
                print("[DEPTH] Model doesn't support max_depth, using as relative")
                use_metric = False
                
        except ImportError as e:
            print(f"[DEPTH] Import error: {e}")
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
                use_metric = False
            except ImportError:
                print("[DEPTH] CRITICAL: Cannot import DepthAnythingV2!")
                sys.exit(1)
    else:
        checkpoint_path = DEPTH_CHECKPOINT_RELATIVE
        if not os.path.exists(checkpoint_path):
            print(f"[DEPTH] ERROR: No checkpoint found!")
            sys.exit(1)
        
        print(f"[DEPTH] Using RELATIVE model: {checkpoint_path}")
        
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            print("[DEPTH] CRITICAL: Cannot import DepthAnythingV2!")
            sys.exit(1)
    
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device).eval()
    
    if device == 'cuda':
        model = model.half()
        print(f"[DEPTH] GPU: {torch.cuda.get_device_name(0)}")
    
    return model, device, use_metric


# ==========================================
# METRIC DEPTH ENGINE
# ==========================================
class MetricDepthEngine:
    """GPU Depth Estimation with temporal smoothing."""
    
    def __init__(self):
        print("[DEPTH] Initializing...")
        self.model, self.device, self.is_metric = load_depth_model()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        
        self.depth_history: Deque[np.ndarray] = deque(maxlen=TEMPORAL_SMOOTH_FRAMES)
        print("[DEPTH] Engine ready!")

    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run depth inference, returns (visualization, meters)."""
        h, w = frame.shape[:2]
        
        img = cv2.resize(frame, (518, 518))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) / 255.0
        
        if self.device == 'cuda':
            img_t = img_t.half()
            img_t = (img_t - self.mean.half()) / self.std.half()
        else:
            img_t = (img_t - self.mean) / self.std
        
        img_t = img_t.unsqueeze(0)
        depth = self.model(img_t)
        
        depth = F.interpolate(depth[:, None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.float().cpu().numpy()
        
        if self.is_metric:
            depth_meters = depth.copy()
        else:
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = (depth - d_min) / (d_max - d_min)
                depth_norm = 1.0 - depth_norm
                depth_meters = 0.5 + depth_norm * 5.5
            else:
                depth_meters = np.full_like(depth, 3.0)
        
        depth_meters = np.clip(depth_meters, 0.1, 20.0)
        
        self.depth_history.append(depth_meters.copy())
        if len(self.depth_history) >= 2:
            depth_meters = np.mean(self.depth_history, axis=0)
        
        d_min, d_max = depth_meters.min(), depth_meters.max()
        if d_max - d_min > 0.1:
            depth_vis = ((1.0 - (depth_meters - d_min) / (d_max - d_min)) * 255).astype(np.uint8)
        else:
            depth_vis = np.full_like(depth_meters, 128, dtype=np.uint8)
        
        return depth_vis, depth_meters.astype(np.float32)


# ==========================================
# FLOOR SEGMENTATION ENGINE (OneFormer)
# ==========================================
class FloorSegmentationEngine:
    """
    OneFormer-based floor segmentation for indoor environments.
    Uses ADE20K model to identify walkable floor surfaces.
    """
    
    def __init__(self):
        if not ONEFORMER_AVAILABLE:
            self.enabled = False
            print("[SEG] Floor segmentation disabled - OneFormer not available")
            return
            
        print("[SEG] Initializing OneFormer ADE20K (OFFLINE MODE)...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Load with offline mode - no network requests
            self.processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_tiny",
                local_files_only=True
            )
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_tiny",
                local_files_only=True
            ).to(self.device)
            
            if self.device == 'cuda':
                self.model = self.model.half()
            
            self.model.eval()
            self.enabled = True
            print(f"[SEG] OneFormer ready on {self.device} (OFFLINE MODE)")
            
        except Exception as e:
            print(f"[SEG] Failed to load OneFormer: {e}")
            print("[SEG] Make sure you've pre-downloaded the model while connected to internet:")
            print("      python -c \"from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation; "
                  "OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny'); "
                  "OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')\"")
            self.enabled = False
    
    @torch.inference_mode()
    def segment_floor(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment floor from frame.
        
        Returns:
            floor_mask: Binary mask of floor pixels (H x W)
            seg_vis: Colored visualization of segmentation
        """
        if not self.enabled:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), frame.copy()
        
        h, w = frame.shape[:2]
        
        # Resize for faster inference
        scale = SEG_INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Process for semantic segmentation
        inputs = self.processor(
            images=frame_rgb, 
            task_inputs=["semantic"], 
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        
        if self.device == 'cuda':
            # Convert to half precision
            for k, v in inputs.items():
                if v.dtype == torch.float32:
                    inputs[k] = v.half()
        
        # Run model
        outputs = self.model(**inputs)
        
        # Post-process
        predicted_map = self.processor.post_process_semantic_segmentation(
            outputs, 
            target_sizes=[(new_h, new_w)]
        )[0]
        
        seg_map = predicted_map.cpu().numpy()
        
        # Create floor mask
        floor_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        for class_id in FLOOR_CLASS_IDS.keys():
            floor_mask[seg_map == class_id] = 255
        
        # Resize back to original size
        floor_mask = cv2.resize(floor_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create visualization
        seg_vis = self._create_visualization(seg_map, frame_rgb, new_w, new_h, w, h)
        
        return floor_mask, seg_vis
    
    def _create_visualization(self, seg_map, frame_rgb, new_w, new_h, w, h):
        """Create colored visualization of segmentation."""
        # Color map for floor classes
        colors = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        # Floor = Green
        for class_id in FLOOR_CLASS_IDS.keys():
            colors[seg_map == class_id] = [0, 255, 0]
        
        # Walls = Blue (class 0)
        colors[seg_map == 0] = [255, 100, 100]
        
        # Ceiling = Purple (class 5)
        colors[seg_map == 5] = [255, 0, 255]
        
        # Blend with original image
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        blended = cv2.addWeighted(frame_resized, 0.5, colors, 0.5, 0)
        
        # Resize back
        blended = cv2.resize(blended, (w, h))
        return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


# ==========================================
# PATH PROJECTION ENGINE
# ==========================================
class PathProjectionEngine:
    """
    Projects 3D world path onto 2D camera view.
    Used to visualize walkable path for humans on the camera feed.
    """
    
    def __init__(self):
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DIST_COEFFS
        
    def world_to_camera(self, world_points: np.ndarray, 
                        drone_x: float, drone_y: float, drone_z: float, 
                        drone_yaw: float) -> np.ndarray:
        """
        Transform world coordinates to camera coordinates.
        """
        # Translate to drone-centric coordinates
        points = world_points.copy()
        points[:, 0] -= drone_x
        points[:, 1] -= drone_y
        points[:, 2] -= drone_z
        
        # Rotate by negative yaw (world to camera)
        yaw_rad = np.radians(drone_yaw)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Rotation matrix (around Y axis for yaw)
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        
        # Transform to camera frame
        camera_points = np.zeros_like(points)
        for i, p in enumerate(points):
            rotated = R @ p
            camera_points[i, 0] = rotated[0]
            camera_points[i, 1] = -rotated[1]
            camera_points[i, 2] = rotated[2]
        
        return camera_points
    
    def project_to_image(self, camera_points: np.ndarray, 
                         image_width: int = 960, 
                         image_height: int = 720) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project camera coordinates to image coordinates.
        """
        valid_mask = camera_points[:, 2] > 0.1
        
        image_points = np.zeros((len(camera_points), 2))
        
        for i, (x, y, z) in enumerate(camera_points):
            if z > 0.1:
                u = self.camera_matrix[0, 0] * x / z + self.camera_matrix[0, 2]
                v = self.camera_matrix[1, 1] * y / z + self.camera_matrix[1, 2]
                
                if 0 <= u < image_width and 0 <= v < image_height:
                    image_points[i] = [u, v]
                else:
                    valid_mask[i] = False
            else:
                valid_mask[i] = False
        
        return image_points, valid_mask
    
    def project_path(self, world_path: List[Tuple[float, float]], 
                     drone_x: float, drone_y: float, drone_z: float,
                     drone_yaw: float,
                     floor_height: float = 0.0,
                     image_width: int = 960,
                     image_height: int = 720) -> Tuple[List[Tuple[int, int]], List[bool]]:
        """
        Project a 2D floor path to image coordinates.
        """
        if not world_path:
            return [], []
        
        # Convert 2D path to 3D (add Y coordinate = floor height)
        world_points_3d = np.array([
            [wx, floor_height, wz] for wx, wz in world_path
        ])
        
        # Transform to camera coordinates
        camera_points = self.world_to_camera(
            world_points_3d, drone_x, drone_y, drone_z, drone_yaw
        )
        
        # Project to image
        image_points, valid_mask = self.project_to_image(
            camera_points, image_width, image_height
        )
        
        # Convert to integer coordinates
        image_points_int = [(int(p[0]), int(p[1])) for p in image_points]
        
        return image_points_int, valid_mask.tolist()
    
    def draw_path_on_image(self, image: np.ndarray,
                           image_points: List[Tuple[int, int]],
                           valid_mask: List[bool],
                           start_idx: int = 0,
                           goal_idx: int = -1) -> np.ndarray:
        """
        Draw projected path on image with visual enhancements.
        """
        if not image_points:
            return image
        
        result = image.copy()
        
        # Get valid points
        valid_points = [(i, p) for i, (p, v) in enumerate(zip(image_points, valid_mask)) if v]
        
        if not valid_points:
            return result
        
        # Draw path lines connecting valid points
        for i in range(len(valid_points) - 1):
            idx1, pt1 = valid_points[i]
            idx2, pt2 = valid_points[i + 1]
            
            # Draw gradient line (closer = thicker)
            thickness = max(2, min(8, int(8 - i * 0.5)))
            cv2.line(result, pt1, pt2, PATH_LINE_COLOR, thickness)
        
        # Draw waypoints
        for i, (idx, pt) in enumerate(valid_points):
            radius = max(3, min(10, int(10 - i * 0.3)))
            cv2.circle(result, pt, radius, PATH_POINT_COLOR, -1)
            cv2.circle(result, pt, radius + 2, (255, 255, 255), 1)
        
        # Draw start marker
        if valid_points:
            _, start_pt = valid_points[0]
            cv2.circle(result, start_pt, PATH_POINT_RADIUS + 4, PATH_START_COLOR, -1)
            cv2.putText(result, "START", (start_pt[0] + 15, start_pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw goal marker
        if len(valid_points) > 1:
            _, goal_pt = valid_points[-1]
            cv2.circle(result, goal_pt, PATH_POINT_RADIUS + 4, PATH_GOAL_COLOR, -1)
            cv2.putText(result, "GOAL", (goal_pt[0] + 15, goal_pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result


# ==========================================
# V17 IMPROVED A* PATHFINDING WITH STRING PULLING + B-SPLINE
# ==========================================
class AStarPathfinder:
    """
    A* pathfinding algorithm with V17 improvements:
    1. A* search for initial path
    2. String Pulling: Raycast from A to C; if clear, delete B
    3. B-Spline smoothing for rounded corners
    """
    
    def __init__(self, grid_size: int = FLOOR_GRID_SIZE):
        self.grid_size = grid_size
        self.center = grid_size // 2
        
        # 8-directional movement
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        self.direction_costs = [
            1.0, 1.0, 1.0, 1.0,
            1.414, 1.414, 1.414, 1.414
        ]
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def find_path(self, floor_grid: np.ndarray, start: Tuple[int, int], 
                  goal: Tuple[int, int], obstacle_grid: np.ndarray = None) -> List[Tuple[int, int]]:
        """
        Find path from start to goal on floor grid.
        V17: Now includes string pulling and B-spline smoothing.
        """
        if not self._is_valid(start, floor_grid, obstacle_grid):
            start = self._find_nearest_valid(start, floor_grid, obstacle_grid)
            if start is None:
                return []
        
        if not self._is_valid(goal, floor_grid, obstacle_grid):
            goal = self._find_nearest_valid(goal, floor_grid, obstacle_grid)
            if goal is None:
                return []
        
        # Step 1: Run A* to get initial path
        raw_path = self._astar_search(floor_grid, start, goal, obstacle_grid)
        
        if not raw_path:
            return []
        
        # Step 2: String Pulling - Remove unnecessary waypoints
        if PATH_STRING_PULL_ENABLED and len(raw_path) > 2:
            pulled_path = self._string_pull(raw_path, floor_grid, obstacle_grid)
        else:
            pulled_path = raw_path
        
        # Step 3: B-Spline smoothing for rounded corners
        if PATH_BSPLINE_ENABLED and SCIPY_AVAILABLE and len(pulled_path) > 3:
            smoothed_path = self._bspline_smooth(pulled_path)
        else:
            smoothed_path = pulled_path
        
        return smoothed_path
    
    def _astar_search(self, floor_grid: np.ndarray, start: Tuple[int, int],
                      goal: Tuple[int, int], obstacle_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Core A* search algorithm."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self.heuristic(start, goal)}
        
        open_set_hash: Set[Tuple[int, int]] = {start}
        
        max_iterations = self.grid_size * self.grid_size
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for i, (dx, dz) in enumerate(self.directions):
                neighbor = (current[0] + dx, current[1] + dz)
                
                if not self._is_valid(neighbor, floor_grid, obstacle_grid):
                    continue
                
                tentative_g = g_score[current] + self.direction_costs[i]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return []
    
    def _string_pull(self, path: List[Tuple[int, int]], floor_grid: np.ndarray,
                     obstacle_grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        V17: String Pulling Algorithm
        Raycast from point A to C; if clear, remove B.
        This removes unnecessary zig-zags from the path.
        """
        if len(path) <= 2:
            return path
        
        pulled = [path[0]]  # Start with first point
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to find the furthest point we can see directly
            furthest_visible = current_idx + 1
            
            for check_idx in range(len(path) - 1, current_idx + 1, -1):
                if self._line_of_sight(path[current_idx], path[check_idx], 
                                       floor_grid, obstacle_grid):
                    furthest_visible = check_idx
                    break
            
            pulled.append(path[furthest_visible])
            current_idx = furthest_visible
        
        return pulled
    
    def _bspline_smooth(self, path: List[Tuple[int, int]], 
                        num_samples: int = PATH_BSPLINE_SAMPLES) -> List[Tuple[int, int]]:
        """
        V17: B-Spline smoothing using scipy
        Rounds sharp corners for smoother drone movement.
        """
        if len(path) < 4:
            return path
        
        try:
            # Extract x and z coordinates
            x = np.array([p[0] for p in path], dtype=float)
            z = np.array([p[1] for p in path], dtype=float)
            
            # Create parametric representation
            # Use splprep for parametric splines
            tck, u = interpolate.splprep([x, z], s=PATH_BSPLINE_SMOOTHNESS, k=min(3, len(path)-1))
            
            # Evaluate spline at many points for smooth curve
            u_new = np.linspace(0, 1, num_samples)
            smooth_coords = interpolate.splev(u_new, tck)
            
            # Convert back to integer grid coordinates
            smoothed_path = []
            for sx, sz in zip(smooth_coords[0], smooth_coords[1]):
                # Clamp to grid bounds
                gx = int(np.clip(round(sx), 0, self.grid_size - 1))
                gz = int(np.clip(round(sz), 0, self.grid_size - 1))
                
                # Avoid duplicate consecutive points
                if not smoothed_path or (gx, gz) != smoothed_path[-1]:
                    smoothed_path.append((gx, gz))
            
            # Ensure start and end points are preserved exactly
            if smoothed_path and smoothed_path[0] != path[0]:
                smoothed_path[0] = path[0]
            if smoothed_path and smoothed_path[-1] != path[-1]:
                smoothed_path[-1] = path[-1]
            
            return smoothed_path
            
        except Exception as e:
            print(f"[PATH] B-spline smoothing failed: {e}")
            return path
    
    def _is_valid(self, pos: Tuple[int, int], floor_grid: np.ndarray, 
                  obstacle_grid: np.ndarray = None) -> bool:
        x, z = pos
        
        if x < 0 or x >= self.grid_size or z < 0 or z >= self.grid_size:
            return False
        
        if floor_grid[z, x] <= 0:
            return False
        
        if obstacle_grid is not None and obstacle_grid[z, x] > 0:
            return False
        
        return True
    
    def _find_nearest_valid(self, pos: Tuple[int, int], floor_grid: np.ndarray,
                           obstacle_grid: np.ndarray = None, max_radius: int = 20) -> Optional[Tuple[int, int]]:
        for r in range(1, max_radius):
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) != r and abs(dz) != r:
                        continue
                    candidate = (pos[0] + dx, pos[1] + dz)
                    if self._is_valid(candidate, floor_grid, obstacle_grid):
                        return candidate
        return None
    
    def _line_of_sight(self, p1: Tuple[int, int], p2: Tuple[int, int],
                       floor_grid: np.ndarray, obstacle_grid: np.ndarray = None) -> bool:
        """
        Bresenham's line algorithm to check line of sight.
        Returns True if there's a clear path from p1 to p2.
        """
        x1, z1 = p1
        x2, z2 = p2
        
        dx = abs(x2 - x1)
        dz = abs(z2 - z1)
        
        sx = 1 if x1 < x2 else -1
        sz = 1 if z1 < z2 else -1
        
        err = dx - dz
        
        while True:
            if not self._is_valid((x1, z1), floor_grid, obstacle_grid):
                return False
            
            if x1 == x2 and z1 == z2:
                break
            
            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x1 += sx
            if e2 < dx:
                err += dx
                z1 += sz
        
        return True


# ==========================================
# 3D FLOOR MAP BUILDER
# ==========================================
class FloorMap3D:
    """
    Builds a 3D map of the floor from depth + segmentation.
    """
    
    def __init__(self):
        self.floor_grid = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE), dtype=np.float32)
        self.floor_confidence = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE), dtype=np.float32)
        self.obstacle_grid = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE), dtype=np.float32)
        
        self.floor_points: List[np.ndarray] = []
        self.floor_colors: List[np.ndarray] = []
        self.scene_points: List[np.ndarray] = []
        self.scene_colors: List[np.ndarray] = []
        
        self.target_position: Optional[Tuple[int, int]] = None
        self.start_position: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []
        self.world_path: List[Tuple[float, float]] = []
        
        self.pathfinder = AStarPathfinder(FLOOR_GRID_SIZE)
        self.lock = threading.Lock()
        
        print("[FLOOR MAP] 3D Floor map initialized with V17 pathfinding")
    
    def update(self, depth_meters: np.ndarray, floor_mask: np.ndarray,
               pos_x: float, pos_y: float, pos_z: float, yaw: float):
        """Update floor map with new depth and segmentation data."""
        h, w = depth_meters.shape
        
        # Set start position if not set
        if self.start_position is None:
            self.start_position = (
                int(pos_x / FLOOR_VOXEL_RES + FLOOR_CENTER),
                int(pos_z / FLOOR_VOXEL_RES + FLOOR_CENTER)
            )
        
        ds = 8
        depth_ds = cv2.resize(depth_meters, (w // ds, h // ds), interpolation=cv2.INTER_NEAREST)
        mask_ds = cv2.resize(floor_mask, (w // ds, h // ds), interpolation=cv2.INTER_NEAREST)
        
        dh, dw = depth_ds.shape
        
        u = np.arange(dw) * ds + ds // 2
        v = np.arange(dh) * ds + ds // 2
        uu, vv = np.meshgrid(u, v)
        
        z = depth_ds.flatten()
        x = (uu.flatten() - CX) * z / FX
        y = (vv.flatten() - CY) * z / FY
        is_floor = mask_ds.flatten() > 0
        
        valid = (z > 0.3) & (z < 6.0)
        x, y, z, is_floor = x[valid], y[valid], z[valid], is_floor[valid]
        
        if len(x) == 0:
            return
        
        yaw_rad = np.radians(-yaw)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        
        x_w = x * c - z * s + pos_x
        z_w = x * s + z * c + pos_z
        y_w = pos_y - y
        
        with self.lock:
            floor_x = x_w[is_floor]
            floor_z = z_w[is_floor]
            floor_y = y_w[is_floor]
            
            floor_height_valid = floor_y < 0.3
            floor_x = floor_x[floor_height_valid]
            floor_z = floor_z[floor_height_valid]
            floor_y = floor_y[floor_height_valid]
            
            ix = (floor_x / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            iz = (floor_z / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            
            valid_grid = ((ix >= 0) & (ix < FLOOR_GRID_SIZE) & 
                         (iz >= 0) & (iz < FLOOR_GRID_SIZE))
            
            ix, iz = ix[valid_grid], iz[valid_grid]
            
            for i, j in zip(ix, iz):
                self.floor_grid[j, i] = 1.0
                self.floor_confidence[j, i] = min(self.floor_confidence[j, i] + 0.1, 1.0)
            
            obs_x = x_w[~is_floor]
            obs_z = z_w[~is_floor]
            obs_y = y_w[~is_floor]
            
            obs_height_valid = (obs_y > 0.1) & (obs_y < 2.0)
            obs_x = obs_x[obs_height_valid]
            obs_z = obs_z[obs_height_valid]
            obs_y_filtered = obs_y[obs_height_valid]
            
            obs_ix = (obs_x / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            obs_iz = (obs_z / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            
            valid_obs = ((obs_ix >= 0) & (obs_ix < FLOOR_GRID_SIZE) & 
                        (obs_iz >= 0) & (obs_iz < FLOOR_GRID_SIZE))
            
            for i, j in zip(obs_ix[valid_obs], obs_iz[valid_obs]):
                self.obstacle_grid[j, i] = min(self.obstacle_grid[j, i] + 0.1, 1.0)
            
            if len(floor_x) > 0:
                floor_x_valid = floor_x[valid_grid[:len(floor_x)] if len(valid_grid) >= len(floor_x) else valid_grid]
                floor_z_valid = floor_z[valid_grid[:len(floor_z)] if len(valid_grid) >= len(floor_z) else valid_grid]
                floor_y_valid = floor_y[valid_grid[:len(floor_y)] if len(valid_grid) >= len(floor_y) else valid_grid]
                
                if len(floor_x_valid) > 0:
                    new_points = np.column_stack([floor_x_valid, floor_y_valid, floor_z_valid])
                    new_colors = np.full((len(new_points), 3), [0.2, 0.8, 0.2])
                    
                    self.floor_points.append(new_points[:500])
                    self.floor_colors.append(new_colors[:500])
            
            if len(obs_x) > 0 and len(valid_obs) > 0:
                obs_x_valid = obs_x[valid_obs]
                obs_z_valid = obs_z[valid_obs]
                obs_y_valid = obs_y_filtered[valid_obs]
                
                if len(obs_x_valid) > 0:
                    scene_pts = np.column_stack([obs_x_valid, obs_y_valid, obs_z_valid])
                    scene_cols = np.full((len(scene_pts), 3), [0.8, 0.3, 0.3])
                    
                    self.scene_points.append(scene_pts[:200])
                    self.scene_colors.append(scene_cols[:200])
            
            if len(self.floor_points) > 100:
                self.floor_points = self.floor_points[-50:]
                self.floor_colors = self.floor_colors[-50:]
            
            if len(self.scene_points) > 50:
                self.scene_points = self.scene_points[-25:]
                self.scene_colors = self.scene_colors[-25:]
    
    def set_target(self, world_x: float, world_z: float):
        """Set target position in world coordinates."""
        ix = int(world_x / FLOOR_VOXEL_RES + FLOOR_CENTER)
        iz = int(world_z / FLOOR_VOXEL_RES + FLOOR_CENTER)
        self.target_position = (ix, iz)
        print(f"[FLOOR MAP] Target set at grid ({ix}, {iz})")
    
    def find_path(self, drone_x: float, drone_z: float) -> List[Tuple[float, float]]:
        """Find path from drone position to target."""
        if self.target_position is None:
            return []
        
        if self.start_position is not None:
            start_x, start_z = self.start_position
        else:
            start_x = int(drone_x / FLOOR_VOXEL_RES + FLOOR_CENTER)
            start_z = int(drone_z / FLOOR_VOXEL_RES + FLOOR_CENTER)
        
        with self.lock:
            grid_path = self.pathfinder.find_path(
                self.floor_grid,
                (start_x, start_z),
                self.target_position,
                (self.obstacle_grid > 0.5).astype(np.float32)
            )
        
        if not grid_path:
            return []
        
        world_path = []
        for gx, gz in grid_path:
            wx = (gx - FLOOR_CENTER) * FLOOR_VOXEL_RES
            wz = (gz - FLOOR_CENTER) * FLOOR_VOXEL_RES
            world_path.append((wx, wz))
        
        self.current_path = grid_path
        self.world_path = world_path
        return world_path
    
    def get_floor_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock:
            if not self.floor_points:
                return np.zeros((0, 3)), np.zeros((0, 3))
            
            points = np.vstack(self.floor_points)
            colors = np.vstack(self.floor_colors)
            
            return points, colors
    
    def get_scene_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock:
            if not self.scene_points:
                return np.zeros((0, 3)), np.zeros((0, 3))
            
            points = np.vstack(self.scene_points)
            colors = np.vstack(self.scene_colors)
            
            return points, colors
    
    def get_visualization_2d(self) -> np.ndarray:
        vis = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE, 3), dtype=np.uint8)
        
        with self.lock:
            vis[self.floor_grid > 0] = [0, 150, 0]
            vis[self.floor_confidence > 0.5] = [0, 255, 0]
            vis[self.obstacle_grid > 0.5] = [0, 0, 200]
            
            for gx, gz in self.current_path:
                if 0 <= gx < FLOOR_GRID_SIZE and 0 <= gz < FLOOR_GRID_SIZE:
                    cv2.circle(vis, (gx, gz), 2, (0, 255, 255), -1)
            
            if self.start_position:
                sx, sz = self.start_position
                if 0 <= sx < FLOOR_GRID_SIZE and 0 <= sz < FLOOR_GRID_SIZE:
                    cv2.circle(vis, (sx, sz), 5, (255, 255, 0), -1)
            
            if self.target_position:
                tx, tz = self.target_position
                if 0 <= tx < FLOOR_GRID_SIZE and 0 <= tz < FLOOR_GRID_SIZE:
                    cv2.circle(vis, (tx, tz), 5, (255, 0, 0), -1)
        
        return vis


# ==========================================
# OPEN3D 3D VISUALIZATION THREAD
# ==========================================
class Open3DVisualizer:
    """Real-time 3D visualization of floor map using Open3D."""
    
    def __init__(self, floor_map: FloorMap3D):
        if not OPEN3D_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        self.floor_map = floor_map
        self.running = False
        
        self.vis = None
        self.pcd_floor = None
        self.pcd_scene = None
        self.path_lines = None
        self.drone_marker = None
        self.target_marker = None
        self.start_marker = None
        
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.drone_yaw = 0
        
        self.lock = threading.Lock()
        self.first_update = True
        
    def start(self):
        if not self.enabled:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[3D VIS] Open3D visualization started")
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
    
    def update_drone_position(self, x: float, y: float, z: float, yaw: float):
        with self.lock:
            self.drone_pos = np.array([x, y, z])
            self.drone_yaw = yaw
    
    def _run(self):
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("Floor Map 3D - V17", width=1024, height=768)
            
            self.pcd_floor = o3d.geometry.PointCloud()
            self.pcd_scene = o3d.geometry.PointCloud()
            self.path_lines = o3d.geometry.LineSet()
            
            self.drone_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            self.drone_marker.paint_uniform_color([1, 0, 0])
            self.drone_marker.compute_vertex_normals()
            
            self.target_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            self.target_marker.paint_uniform_color([0, 0, 1])
            self.target_marker.compute_vertex_normals()
            
            self.start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            self.start_marker.paint_uniform_color([0, 1, 1])
            self.start_marker.compute_vertex_normals()
            
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            ground_grid = self._create_ground_grid()
            
            self.vis.add_geometry(self.pcd_floor)
            self.vis.add_geometry(self.pcd_scene)
            self.vis.add_geometry(self.path_lines)
            self.vis.add_geometry(self.drone_marker)
            self.vis.add_geometry(self.target_marker)
            self.vis.add_geometry(self.start_marker)
            self.vis.add_geometry(coord_frame)
            self.vis.add_geometry(ground_grid)
            
            opt = self.vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.15])
            opt.point_size = 3.0
            
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.4)
            ctr.set_front([0, -0.5, -0.5])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 1, 0])
            
            last_update = time.time()
            
            while self.running:
                if time.time() - last_update > 0.1:
                    self._update_geometries()
                    last_update = time.time()
                
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.01)
            
            self.vis.destroy_window()
            
        except Exception as e:
            print(f"[3D VIS] Error: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False
    
    def _create_ground_grid(self):
        lines = []
        points = []
        grid_size = 10
        grid_step = 1.0
        
        idx = 0
        for i in range(-grid_size, grid_size + 1):
            x = i * grid_step
            points.append([x, 0, -grid_size * grid_step])
            points.append([x, 0, grid_size * grid_step])
            lines.append([idx, idx + 1])
            idx += 2
            
            points.append([-grid_size * grid_step, 0, x])
            points.append([grid_size * grid_step, 0, x])
            lines.append([idx, idx + 1])
            idx += 2
        
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.paint_uniform_color([0.3, 0.3, 0.3])
        
        return grid
    
    def _update_geometries(self):
        try:
            floor_points, floor_colors = self.floor_map.get_floor_point_cloud()
            
            if len(floor_points) > 0:
                self.pcd_floor.points = o3d.utility.Vector3dVector(floor_points)
                self.pcd_floor.colors = o3d.utility.Vector3dVector(floor_colors)
                self.vis.update_geometry(self.pcd_floor)
            
            scene_points, scene_colors = self.floor_map.get_scene_point_cloud()
            
            if len(scene_points) > 0:
                self.pcd_scene.points = o3d.utility.Vector3dVector(scene_points)
                self.pcd_scene.colors = o3d.utility.Vector3dVector(scene_colors)
                self.vis.update_geometry(self.pcd_scene)
            
            with self.lock:
                drone_pos = self.drone_pos.copy()
            
            self.drone_marker.translate(-self.drone_marker.get_center())
            self.drone_marker.translate(drone_pos)
            self.vis.update_geometry(self.drone_marker)
            
            if self.floor_map.target_position:
                tx, tz = self.floor_map.target_position
                wx = (tx - FLOOR_CENTER) * FLOOR_VOXEL_RES
                wz = (tz - FLOOR_CENTER) * FLOOR_VOXEL_RES
                self.target_marker.translate(-self.target_marker.get_center())
                self.target_marker.translate([wx, 0.1, wz])
                self.vis.update_geometry(self.target_marker)
            
            if self.floor_map.start_position:
                sx, sz = self.floor_map.start_position
                wx = (sx - FLOOR_CENTER) * FLOOR_VOXEL_RES
                wz = (sz - FLOOR_CENTER) * FLOOR_VOXEL_RES
                self.start_marker.translate(-self.start_marker.get_center())
                self.start_marker.translate([wx, 0.1, wz])
                self.vis.update_geometry(self.start_marker)
            
            if len(self.floor_map.current_path) > 1:
                path_points = []
                for gx, gz in self.floor_map.current_path:
                    wx = (gx - FLOOR_CENTER) * FLOOR_VOXEL_RES
                    wz = (gz - FLOOR_CENTER) * FLOOR_VOXEL_RES
                    path_points.append([wx, 0.05, wz])
                
                lines = [[i, i+1] for i in range(len(path_points)-1)]
                colors = [[1, 1, 0] for _ in lines]
                
                self.path_lines.points = o3d.utility.Vector3dVector(path_points)
                self.path_lines.lines = o3d.utility.Vector2iVector(lines)
                self.path_lines.colors = o3d.utility.Vector3dVector(colors)
                self.vis.update_geometry(self.path_lines)
                
        except Exception as e:
            print(f"[3D VIS] Update error: {e}")


# ==========================================
# V17 ENHANCED OBSTACLE DETECTOR WITH ADAPTIVE SPEED
# ==========================================
@dataclass
class ObstacleInfo:
    """Structured obstacle detection with V17 adaptive speed support."""
    center_dist: float = 10.0
    left_dist: float = 10.0
    right_dist: float = 10.0
    top_dist: float = 10.0
    bottom_dist: float = 10.0
    min_dist: float = 10.0
    
    column_dists: List[float] = field(default_factory=lambda: [10.0] * THIN_OBSTACLE_COLUMNS)
    
    thin_obstacle_detected: bool = False
    thin_obstacle_column: int = -1
    
    is_emergency: bool = False
    is_critical: bool = False
    is_danger: bool = False
    is_caution: bool = False
    is_clear: bool = True
    
    is_corner: bool = False
    corner_type: str = ""
    
    # V17: Ceiling detection
    ceiling_detected: bool = False
    ceiling_dist: float = 10.0
    
    safe_direction: str = "forward"
    steer_amount: float = 0.0
    can_go_up: bool = True
    can_go_down: bool = True
    
    escape_rotation: int = 0
    
    # V17: Veering recommendation
    veer_direction: str = "none"  # "left", "right", "none"
    veer_strength: float = 0.0   # 0.0 to 1.0
    
    # V17: Adaptive speed mode
    speed_mode: SpeedMode = SpeedMode.CRUISE
    recommended_speed: int = 30


class EnhancedObstacleDetector:
    """
    V17: Multi-zone obstacle detection with adaptive speed control.
    Automatically adjusts speed based on obstacle proximity.
    """
    
    def __init__(self):
        self.history: Deque[ObstacleInfo] = deque(maxlen=5)
        self.emergency_count = 0
        self.corner_frames = 0
    
    def _detect_thin_obstacles(self, depth_meters: np.ndarray) -> Tuple[bool, int, float]:
        h, w = depth_meters.shape
        
        depth_uint8 = ((depth_meters - depth_meters.min()) / 
                       (depth_meters.max() - depth_meters.min() + 1e-6) * 255).astype(np.uint8)
        
        sobel_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        center_region = edges[int(h*0.2):int(h*0.8), int(w*0.15):int(w*0.85)]
        depth_center = depth_meters[int(h*0.2):int(h*0.8), int(w*0.15):int(w*0.85)]
        
        col_width = center_region.shape[1] // THIN_OBSTACLE_COLUMNS
        
        thin_detected = False
        thin_column = -1
        min_thin_dist = 10.0
        
        for i in range(THIN_OBSTACLE_COLUMNS):
            col_start = i * col_width
            col_end = (i + 1) * col_width
            
            col_edges = center_region[:, col_start:col_end]
            col_depth = depth_center[:, col_start:col_end]
            
            edge_strength = np.mean(col_edges)
            
            if edge_strength > EDGE_DETECTION_THRESHOLD:
                col_min = np.percentile(col_depth, 5)
                col_max = np.percentile(col_depth, 95)
                
                if col_max - col_min > THIN_OBSTACLE_THRESHOLD and col_min < DANGER_DIST:
                    thin_detected = True
                    if col_min < min_thin_dist:
                        min_thin_dist = col_min
                        thin_column = i
        
        return thin_detected, thin_column, min_thin_dist
    
    def _detect_corner(self, left_dist: float, right_dist: float, 
                       center_dist: float) -> Tuple[bool, str]:
        left_blocked = left_dist < CORNER_LEFT_THRESHOLD
        right_blocked = right_dist < CORNER_RIGHT_THRESHOLD
        front_blocked = center_dist < CORNER_FRONT_THRESHOLD
        
        if front_blocked and left_blocked and right_blocked:
            return True, "dead_end"
        elif front_blocked and left_blocked:
            return True, "left_corner"
        elif front_blocked and right_blocked:
            return True, "right_corner"
        
        return False, ""
    
    def _compute_veering(self, column_dists: List[float], left_dist: float, 
                         right_dist: float) -> Tuple[str, float]:
        """
        Compute veering direction and strength for smooth obstacle avoidance.
        Returns (direction, strength) where direction is "left", "right", or "none"
        and strength is 0.0 to 1.0.
        """
        # Find best path through columns
        left_cols = column_dists[:THIN_OBSTACLE_COLUMNS // 3]
        center_cols = column_dists[THIN_OBSTACLE_COLUMNS // 3: 2 * THIN_OBSTACLE_COLUMNS // 3]
        right_cols = column_dists[2 * THIN_OBSTACLE_COLUMNS // 3:]
        
        left_avg = np.mean(left_cols) if left_cols else 10.0
        center_avg = np.mean(center_cols) if center_cols else 10.0
        right_avg = np.mean(right_cols) if right_cols else 10.0
        
        # If center is clear, no veering needed
        if center_avg > CAUTION_DIST:
            return "none", 0.0
        
        # Determine best direction
        if left_avg > right_avg and left_avg > center_avg:
            strength = min(1.0, (CAUTION_DIST - center_avg) / CAUTION_DIST)
            return "left", strength
        elif right_avg > left_avg and right_avg > center_avg:
            strength = min(1.0, (CAUTION_DIST - center_avg) / CAUTION_DIST)
            return "right", strength
        
        # Use zone distances as backup
        if left_dist > right_dist:
            strength = min(1.0, (CAUTION_DIST - center_avg) / CAUTION_DIST)
            return "left", strength
        else:
            strength = min(1.0, (CAUTION_DIST - center_avg) / CAUTION_DIST)
            return "right", strength
    
    def _compute_adaptive_speed(self, center_dist: float, min_dist: float) -> Tuple[SpeedMode, int]:
        """
        V17: Compute adaptive speed based on obstacle proximity.
        Returns (speed_mode, recommended_speed).
        """
        # Use the more restrictive of center and min distance
        effective_dist = min(center_dist, min_dist * 1.2)
        
        if effective_dist < EMERGENCY_STOP_DIST:
            return SpeedMode.STOP, SPEED_VALUES[SpeedMode.STOP]
        elif effective_dist < CRITICAL_DIST:
            return SpeedMode.CREEP, SPEED_VALUES[SpeedMode.CREEP]
        elif effective_dist < DANGER_DIST:
            return SpeedMode.SLOW, SPEED_VALUES[SpeedMode.SLOW]
        elif effective_dist < CAUTION_DIST:
            # Interpolate between SLOW and CRUISE
            factor = (effective_dist - DANGER_DIST) / (CAUTION_DIST - DANGER_DIST)
            speed = int(SPEED_VALUES[SpeedMode.SLOW] + 
                       factor * (SPEED_VALUES[SpeedMode.CRUISE] - SPEED_VALUES[SpeedMode.SLOW]))
            return SpeedMode.CRUISE, speed
        elif effective_dist < SAFE_DIST:
            return SpeedMode.CRUISE, SPEED_VALUES[SpeedMode.CRUISE]
        elif effective_dist < CLEAR_DIST:
            # Interpolate between CRUISE and FAST
            factor = (effective_dist - SAFE_DIST) / (CLEAR_DIST - SAFE_DIST)
            speed = int(SPEED_VALUES[SpeedMode.CRUISE] + 
                       factor * (SPEED_VALUES[SpeedMode.FAST] - SPEED_VALUES[SpeedMode.CRUISE]))
            return SpeedMode.FAST, speed
        else:
            return SpeedMode.FAST, SPEED_VALUES[SpeedMode.FAST]
    
    def analyze(self, depth_meters: np.ndarray, current_altitude: float = 1.0) -> ObstacleInfo:
        """Analyze depth map for obstacles with V17 adaptive speed."""
        h, w = depth_meters.shape
        info = ObstacleInfo()
        
        zones = {
            'center': (0.25, 0.75, 0.30, 0.70),
            'center_wide': (0.20, 0.80, 0.20, 0.80),
            'left': (0.25, 0.75, 0.0, 0.35),
            'right': (0.25, 0.75, 0.65, 1.0),
            'top': (0.0, 0.25, 0.20, 0.80),  # Expanded for ceiling
            'bottom': (0.75, 1.0, 0.25, 0.75),
        }
        
        zone_dists = {}
        
        for name, (y1f, y2f, x1f, x2f) in zones.items():
            y1, y2 = int(y1f * h), int(y2f * h)
            x1, x2 = int(x1f * w), int(x2f * w)
            
            zone = depth_meters[y1:y2, x1:x2]
            if zone.size > 0:
                min_dist = float(np.percentile(zone, 5))
                p25_dist = float(np.percentile(zone, 25))
                zone_dists[name] = min_dist * 0.6 + p25_dist * 0.4
            else:
                zone_dists[name] = 10.0
        
        col_width = w // THIN_OBSTACLE_COLUMNS
        for i in range(THIN_OBSTACLE_COLUMNS):
            x1 = i * col_width
            x2 = (i + 1) * col_width
            col_region = depth_meters[int(h*0.2):int(h*0.8), x1:x2]
            if col_region.size > 0:
                info.column_dists[i] = float(np.percentile(col_region, 10)) / SAFETY_MARGIN
        
        thin_detected, thin_col, thin_dist = self._detect_thin_obstacles(depth_meters)
        info.thin_obstacle_detected = thin_detected
        info.thin_obstacle_column = thin_col
        
        if thin_detected and thin_col >= 0:
            info.column_dists[thin_col] = min(info.column_dists[thin_col], thin_dist)
        
        for k in zone_dists:
            zone_dists[k] = zone_dists[k] / SAFETY_MARGIN
        
        info.center_dist = min(zone_dists.get('center', 10.0), 
                               zone_dists.get('center_wide', 10.0))
        info.left_dist = zone_dists.get('left', 10.0)
        info.right_dist = zone_dists.get('right', 10.0)
        info.top_dist = zone_dists.get('top', 10.0)
        info.bottom_dist = zone_dists.get('bottom', 10.0)
        
        # V17: Ceiling detection
        info.ceiling_dist = info.top_dist
        info.ceiling_detected = (info.top_dist < CEILING_CAUTION_DIST or 
                                 current_altitude > MAX_ALTITUDE - 0.3)
        
        center_cols = info.column_dists[THIN_OBSTACLE_COLUMNS//3 : 2*THIN_OBSTACLE_COLUMNS//3]
        if center_cols:
            min_center_col = min(center_cols)
            info.center_dist = min(info.center_dist, min_center_col)
        
        info.min_dist = min(min(zone_dists.values()), min(info.column_dists))
        
        info.is_emergency = info.center_dist < EMERGENCY_STOP_DIST or info.min_dist < EMERGENCY_STOP_DIST * 0.8
        info.is_critical = info.center_dist < CRITICAL_DIST
        info.is_danger = info.center_dist < DANGER_DIST
        info.is_caution = info.center_dist < CAUTION_DIST
        info.is_clear = info.center_dist > SAFE_DIST and info.min_dist > CAUTION_DIST
        
        info.is_corner, info.corner_type = self._detect_corner(
            info.left_dist, info.right_dist, info.center_dist)
        
        if info.is_corner:
            self.corner_frames += 1
        else:
            self.corner_frames = max(0, self.corner_frames - 1)
        
        # V17: Strict ceiling-aware vertical limits
        info.can_go_up = (info.top_dist > CEILING_MIN_DIST and 
                          current_altitude < MAX_ALTITUDE - 0.2 and
                          not info.ceiling_detected)
        info.can_go_down = info.bottom_dist > CAUTION_DIST and current_altitude > MIN_ALTITUDE
        
        if info.is_corner or info.is_critical:
            if info.corner_type == "left_corner" or info.left_dist < info.right_dist:
                info.escape_rotation = SPEED_ESCAPE_YAW
            elif info.corner_type == "right_corner" or info.right_dist < info.left_dist:
                info.escape_rotation = -SPEED_ESCAPE_YAW
            else:
                info.escape_rotation = SPEED_ESCAPE_YAW if info.right_dist >= info.left_dist else -SPEED_ESCAPE_YAW
        
        # V17: Compute veering recommendation
        info.veer_direction, info.veer_strength = self._compute_veering(
            info.column_dists, info.left_dist, info.right_dist
        )
        
        # V17: Compute adaptive speed
        info.speed_mode, info.recommended_speed = self._compute_adaptive_speed(
            info.center_dist, info.min_dist
        )
        
        best_col_idx = int(np.argmax(info.column_dists))
        if info.column_dists[best_col_idx] > DANGER_DIST:
            info.steer_amount = (best_col_idx - THIN_OBSTACLE_COLUMNS // 2) / (THIN_OBSTACLE_COLUMNS // 2)
            
            if best_col_idx < THIN_OBSTACLE_COLUMNS // 3:
                info.safe_direction = "left"
            elif best_col_idx > 2 * THIN_OBSTACLE_COLUMNS // 3:
                info.safe_direction = "right"
            else:
                info.safe_direction = "forward"
        else:
            info.safe_direction = "turn"
            info.steer_amount = 1.0 if info.left_dist < info.right_dist else -1.0
        
        if info.is_emergency:
            self.emergency_count += 1
        else:
            self.emergency_count = max(0, self.emergency_count - 1)
        
        self.history.append(info)
        return info
    
    def get_smoothed_center_dist(self) -> float:
        if not self.history:
            return 10.0
        return np.mean([h.center_dist for h in self.history])
    
    def is_persistent_emergency(self) -> bool:
        return self.emergency_count >= 2
    
    def is_stuck_in_corner(self) -> bool:
        return self.corner_frames >= int(CORNER_STUCK_TIME * 30)


# ==========================================
# SIMPLE PID CONTROLLER
# ==========================================
@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    limit: float = 100.0
    
    _integral: float = field(default=0.0, repr=False)
    _prev_error: float = field(default=0.0, repr=False)
    
    def compute(self, error: float, dt: float = 0.033) -> float:
        self._integral = np.clip(self._integral + error * dt, -50, 50)
        derivative = (error - self._prev_error) / max(dt, 0.001)
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, -self.limit, self.limit))
    
    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


# ==========================================
# ENHANCED VISUAL ODOMETRY
# ==========================================
class EnhancedVisualOdometry:
    """Improved visual odometry with IMU fusion."""
    
    def __init__(self):
        self.K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        self.prev_gray = None
        self.prev_points = None
        
        self.pos_x = 0.0
        self.pos_y = 1.0
        self.pos_z = 0.0
        self.yaw = 0.0
        
        self.vel_x = 0.0
        self.vel_z = 0.0
        self.last_time = time.time()
        
        self.pos_history: Deque[Tuple[float, float, float]] = deque(maxlen=100)
        
        self.imu_yaw_offset = 0.0
        self.imu_initialized = False
    
    def update(self, frame: np.ndarray, imu_yaw: float, imu_height: Optional[float] = None):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not self.imu_initialized and abs(imu_yaw) < 360:
            self.imu_yaw_offset = imu_yaw
            self.imu_initialized = True
        
        self.yaw = float(imu_yaw)
        
        if imu_height is not None and imu_height > 0:
            self.pos_y = 0.9 * self.pos_y + 0.1 * imu_height
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=VO_MAX_FEATURES, 
                qualityLevel=VO_QUALITY, minDistance=VO_MIN_DISTANCE
            )
            return
        
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=VO_MAX_FEATURES,
                qualityLevel=VO_QUALITY, minDistance=VO_MIN_DISTANCE
            )
            return
        
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if p1 is not None:
            good_new = p1[status.flatten() == 1]
            good_old = self.prev_points[status.flatten() == 1]
            
            if len(good_new) > 20:
                flow = good_new - good_old
                mag = np.median(np.linalg.norm(flow, axis=1))
                
                if mag > VO_DEADZONE:
                    try:
                        E, mask = cv2.findEssentialMat(
                            good_new, good_old, self.K,
                            method=cv2.RANSAC, prob=0.999, threshold=0.5
                        )
                        
                        if E is not None:
                            _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, self.K, mask=mask)
                            
                            scale = min(0.04 * mag / 10.0, 0.15) * (self.pos_y / 1.0)
                            
                            yaw_rad = np.radians(-self.yaw)
                            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
                            
                            dx = float(t[0, 0]) * scale
                            dz = float(t[2, 0]) * scale
                            
                            dx_w = dx * c - dz * s
                            dz_w = dx * s + dz * c
                            
                            if not np.isnan(dx_w) and not np.isnan(dz_w):
                                self.pos_x += dx_w * 0.8
                                self.pos_z += dz_w * 0.8
                                
                                self.vel_x = dx_w / max(dt, 0.01)
                                self.vel_z = dz_w / max(dt, 0.01)
                    
                    except Exception:
                        pass
                
                self.prev_points = good_new.reshape(-1, 1, 2)
            else:
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray, maxCorners=VO_MAX_FEATURES,
                    qualityLevel=VO_QUALITY, minDistance=VO_MIN_DISTANCE
                )
        
        self.prev_gray = gray
        self.pos_history.append((self.pos_x, self.pos_y, self.pos_z))
    
    def get_position(self) -> Tuple[float, float, float]:
        return (self.pos_x, self.pos_y, self.pos_z)
    
    def reset_position(self):
        self.pos_x = 0.0
        self.pos_y = 1.0
        self.pos_z = 0.0
        self.pos_history.clear()


# ==========================================
# V17 SAFE VOXEL BRAIN WITH ADAPTIVE SPEED
# ==========================================
class SafeVoxelBrain:
    """
    V17: Enhanced brain with:
    1. Adaptive speed control based on obstacle proximity
    2. Strict altitude hold during rotation (no more ceiling crashes)
    3. Improved pathfinding with string pulling + B-spline
    """
    
    def __init__(self, target_obj: str):
        print(f"[BRAIN V17] Initializing for target: {target_obj}")
        
        self.depth_engine = MetricDepthEngine()
        self.obstacle_detector = EnhancedObstacleDetector()
        self.vo = EnhancedVisualOdometry()
        self.floor_engine = FloorSegmentationEngine()
        self.floor_map = FloorMap3D()
        self.path_projector = PathProjectionEngine()
        
        self.vis3d = Open3DVisualizer(self.floor_map) if OPEN3D_AVAILABLE else None
        
        self.yolo = YOLOWorld('yolov8s-world.pt')
        self.yolo.set_classes([target_obj])
        self.yolo.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_obj = target_obj
        
        self.voxels = np.zeros((GRID_SIZE, GRID_SIZE, HEIGHT_LEVELS), dtype=np.int8)
        self.visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.confidence = np.zeros((GRID_SIZE, GRID_SIZE, HEIGHT_LEVELS), dtype=np.float32)
        
        self.pid_yaw = PID(kp=0.8, ki=0.02, kd=0.12, limit=SPEED_YAW_FAST)
        self.pid_vert = PID(kp=0.4, ki=0.01, kd=0.06, limit=SPEED_VERTICAL)  # Reduced for less bobbing
        
        self.mode = "INIT"
        self.sub_mode = ""
        self.escape_state = EscapeState.NONE
        self.escape_start_yaw = 0.0
        self.escape_target_rotation = 0.0
        self.escape_start_time = 0.0  # V18: Track when escape started for timeout
        
        self.last_pos = (0.0, 0.0, 0.0)
        self.last_move_time = time.time()
        self.explore_direction = 1
        
        self.frames_since_obstacle = 0
        self.consecutive_safe_frames = 0
        
        self.seg_frame_count = 0
        self.last_floor_mask = None
        self.last_seg_vis = None
        
        self.target_world_pos = None
        self.current_world_path = []
        self.last_path_update = 0
        
        # V18: Target detection tracking
        self.target_detected = False
        self.last_target_seen_time = 0
        self.target_detection_count = 0
        
        # V17: Path always on
        self.path_overlay_enabled = True
        self.default_target_set = False
        
        # V17: Current altitude tracking and lock
        self.current_altitude = 1.0
        self.target_altitude = 1.0  # For altitude lock during rotation
        self.altitude_locked = False
        self.rotation_start_altitude = 1.0
        
        # V17: Track if we're in rotation mode
        self.is_rotating = False
        self.rotation_start_time = 0
        
        print("[BRAIN V17] Ready - Adaptive Speed + Ceiling Protection + Advanced Pathfinding")

    @property
    def pos_x(self):
        return self.vo.pos_x
    
    @property
    def pos_y(self):
        return self.vo.pos_y
    
    @property
    def pos_z(self):
        return self.vo.pos_z
    
    @property
    def yaw(self):
        return self.vo.yaw

    def start_visualization(self):
        if self.vis3d:
            self.vis3d.start()

    def stop_visualization(self):
        if self.vis3d:
            self.vis3d.stop()

    def update_vo(self, frame: np.ndarray, imu_yaw: float, imu_height: Optional[float] = None):
        self.vo.update(frame, imu_yaw, imu_height)
        
        # Track altitude for ceiling detection
        if imu_height is not None:
            self.current_altitude = imu_height
        
        if self.vis3d:
            self.vis3d.update_drone_position(self.pos_x, self.pos_y, self.pos_z, self.yaw)

    def update_voxels(self, depth_meters: np.ndarray):
        h, w = depth_meters.shape
        
        ds = 16
        depth_ds = cv2.resize(depth_meters, (w // ds, h // ds), interpolation=cv2.INTER_NEAREST)
        
        dh, dw = depth_ds.shape
        
        u = np.arange(dw) * ds + ds // 2
        v = np.arange(dh) * ds + ds // 2
        uu, vv = np.meshgrid(u, v)
        
        z = depth_ds.flatten()
        x = (uu.flatten() - CX) * z / FX
        y = (vv.flatten() - CY) * z / FY
        
        valid = (z > 0.3) & (z < 8.0)
        x, y, z = x[valid], y[valid], z[valid]
        
        if len(x) == 0:
            return
        
        yaw_rad = np.radians(-self.yaw)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        
        x_w = x * c - z * s + self.pos_x
        z_w = x * s + z * c + self.pos_z
        y_w = self.pos_y - y
        
        valid_h = (y_w > 0.1) & (y_w < HEIGHT_LEVELS * VOXEL_RES - 0.1)
        x_w, y_w, z_w = x_w[valid_h], y_w[valid_h], z_w[valid_h]
        
        ix = (x_w / VOXEL_RES + CENTER).astype(np.int32)
        iz = (z_w / VOXEL_RES + CENTER).astype(np.int32)
        iy = (y_w / VOXEL_RES).astype(np.int32)
        
        valid_grid = ((ix >= 0) & (ix < GRID_SIZE) & 
                      (iz >= 0) & (iz < GRID_SIZE) & 
                      (iy >= 0) & (iy < HEIGHT_LEVELS))
        
        ix, iz, iy = ix[valid_grid], iz[valid_grid], iy[valid_grid]
        
        self.voxels[iz, ix, iy] = 1
        self.confidence[iz, ix, iy] = np.minimum(self.confidence[iz, ix, iy] + 0.2, 1.0)
        
        drone_ix = int(self.pos_x / VOXEL_RES) + CENTER
        drone_iz = int(self.pos_z / VOXEL_RES) + CENTER
        if 0 <= drone_ix < GRID_SIZE and 0 <= drone_iz < GRID_SIZE:
            self.visited[drone_iz, drone_ix] = 1

    def update_floor_map(self, depth_meters: np.ndarray, floor_mask: np.ndarray):
        self.floor_map.update(
            depth_meters, floor_mask,
            self.pos_x, self.pos_y, self.pos_z, self.yaw
        )

    def set_target_from_detection(self, target_world_x: float, target_world_z: float):
        self.target_world_pos = (target_world_x, target_world_z)
        self.floor_map.set_target(target_world_x, target_world_z)
        print(f"[PATH] Target set at world pos: ({target_world_x:.2f}, {target_world_z:.2f})")

    def _auto_set_default_target(self):
        """Automatically set a default target if none exists."""
        if self.target_world_pos is None and not self.default_target_set:
            # Set target 5m in front of current position
            target_x = self.pos_x + DEFAULT_TARGET_DISTANCE * np.sin(np.radians(-self.yaw))
            target_z = self.pos_z - DEFAULT_TARGET_DISTANCE * np.cos(np.radians(-self.yaw))
            self.set_target_from_detection(target_x, target_z)
            self.default_target_set = True
            print(f"[PATH] Auto-set default target {DEFAULT_TARGET_DISTANCE}m ahead")

    def update_path(self):
        if time.time() - self.last_path_update < PATH_UPDATE_INTERVAL:
            return
        
        self.last_path_update = time.time()
        
        # Auto-set target if needed
        self._auto_set_default_target()
        
        if self.target_world_pos is None:
            return
        
        self.current_world_path = self.floor_map.find_path(self.pos_x, self.pos_z)

    def get_path_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Always-on path overlay."""
        # Always try to show path
        if not self.floor_map.world_path:
            # Draw "No Path" indicator
            result = frame.copy()
            cv2.putText(result, "No Path - Press T to set target", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return result
        
        image_points, valid_mask = self.path_projector.project_path(
            self.floor_map.world_path,
            self.pos_x, self.pos_y, self.pos_z, self.yaw,
            floor_height=0.0,
            image_width=frame.shape[1],
            image_height=frame.shape[0]
        )
        
        result = self.path_projector.draw_path_on_image(
            frame, image_points, valid_mask
        )
        
        return result

    def _detect_target(self, frame: np.ndarray) -> Optional[dict]:
        results = self.yolo.predict(frame, verbose=False, conf=0.35, half=True)
        
        if not results or len(results[0].boxes) == 0:
            return None
        
        boxes = results[0].boxes
        best = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
        
        return {
            'bbox': (x1, y1, x2, y2),
            'cx': (x1 + x2) / 2 / w,
            'cy': (y1 + y2) / 2 / h,
            'size': (x2 - x1) * (y2 - y1) / (w * h),
            'boxes': boxes
        }

    def _start_rotation(self):
        """V17: Start a rotation with altitude lock."""
        if not self.is_rotating:
            self.is_rotating = True
            self.rotation_start_time = time.time()
            self.rotation_start_altitude = self.current_altitude
            self.target_altitude = self.current_altitude
            self.altitude_locked = ROTATION_ALTITUDE_LOCK
            print(f"[ALT LOCK] Rotation started, locking altitude at {self.target_altitude:.2f}m")
    
    def _end_rotation(self):
        """V17: End rotation and release altitude lock."""
        if self.is_rotating:
            self.is_rotating = False
            self.altitude_locked = False
            print("[ALT LOCK] Rotation ended, releasing altitude lock")

    def _compute_altitude_correction(self, obs: ObstacleInfo) -> int:
        """
        V17: Compute altitude correction with strict ceiling protection.
        During rotation, maintains altitude lock to prevent rising.
        """
        # If altitude locked during rotation, maintain target altitude
        if self.altitude_locked:
            alt_error = self.target_altitude - self.current_altitude
            
            # Only correct if error is significant
            if abs(alt_error) < ALTITUDE_HOLD_TOLERANCE:
                return 0
            
            # Compute correction to maintain altitude
            correction = int(np.clip(alt_error * 50, -SPEED_VERTICAL_SLOW, SPEED_VERTICAL_SLOW))
            return correction
        
        # Check ceiling limit
        if self.current_altitude >= MAX_ALTITUDE:
            return -SPEED_VERTICAL_SLOW  # Force descent
        
        # Check floor limit  
        if self.current_altitude <= MIN_ALTITUDE:
            return SPEED_VERTICAL_SLOW  # Prevent crash
        
        return 0

    def _handle_corner_escape(self, obs: ObstacleInfo) -> Tuple[int, int, int, int]:
        """V18: Corner escape with altitude lock and timeout to prevent infinite spinning."""
        # Start rotation mode
        self._start_rotation()
        
        if self.escape_state == EscapeState.NONE:
            self.escape_state = EscapeState.CORNER_DETECTED
            self.escape_start_yaw = self.yaw
            self.escape_start_time = time.time()  # V18: Record start time
            
            if obs.right_dist > obs.left_dist:
                self.escape_state = EscapeState.ROTATING_RIGHT
                self.escape_target_rotation = 90
            else:
                self.escape_state = EscapeState.ROTATING_LEFT
                self.escape_target_rotation = -90
        
        # V18: Check for timeout - force exit if spinning too long
        escape_duration = time.time() - self.escape_start_time
        if escape_duration > CORNER_ESCAPE_TIMEOUT:
            print(f"[ESCAPE] Timeout after {escape_duration:.1f}s - forcing exit")
            self.escape_state = EscapeState.ESCAPE_COMPLETE
            self._end_rotation()
            self.sub_mode = "TIMEOUT EXIT"
            # Try moving forward slowly even if not clear
            return (0, SPEED_VALUES[SpeedMode.CREEP], 0, 0)
        
        rotation_done = self.yaw - self.escape_start_yaw
        
        while rotation_done > 180:
            rotation_done -= 360
        while rotation_done < -180:
            rotation_done += 360
        
        # V17: Compute altitude correction during rotation
        vert_correction = self._compute_altitude_correction(obs)
        
        # Check if front is now clear enough to exit
        if obs.center_dist > CORNER_ROTATION_THRESHOLD:
            self.escape_state = EscapeState.ESCAPE_COMPLETE
            self._end_rotation()
            self.sub_mode = "ESCAPE COMPLETE"
            return (0, obs.recommended_speed, vert_correction, 0)
        
        # V18: If rotated more than 270 degrees without finding exit, try alternate escape
        if abs(rotation_done) > 270:
            # Try reversing rotation direction
            if self.escape_state == EscapeState.ROTATING_RIGHT:
                self.escape_state = EscapeState.ROTATING_LEFT
                self.sub_mode = "REVERSE ROT"
            else:
                self.escape_state = EscapeState.ROTATING_RIGHT
                self.sub_mode = "REVERSE ROT"
        
        if self.escape_state == EscapeState.ROTATING_RIGHT:
            self.sub_mode = f"ROT R {abs(rotation_done):.0f}° ({escape_duration:.1f}s)"
            return (0, 0, vert_correction, SPEED_ESCAPE_YAW)
        else:
            self.sub_mode = f"ROT L {abs(rotation_done):.0f}° ({escape_duration:.1f}s)"
            return (0, 0, vert_correction, -SPEED_ESCAPE_YAW)

    def _compute_veering_command(self, obs: ObstacleInfo, base_forward: int) -> Tuple[int, int, int, int]:
        """
        V17: Compute veering command with adaptive speed.
        Speed is automatically reduced based on obstacle proximity.
        """
        lateral = 0
        yaw_cmd = 0
        
        if obs.veer_direction == "left":
            # Veer left: negative lateral, positive yaw
            lateral = -int(SPEED_VEER_LATERAL * obs.veer_strength)
            yaw_cmd = int(SPEED_YAW_SLOW * 0.4 * obs.veer_strength)
        elif obs.veer_direction == "right":
            # Veer right: positive lateral, negative yaw
            lateral = int(SPEED_VEER_LATERAL * obs.veer_strength)
            yaw_cmd = -int(SPEED_YAW_SLOW * 0.4 * obs.veer_strength)
        
        # V17: Use adaptive speed instead of fixed reduction
        # The base_forward is already adapted to obstacle distance
        forward = int(base_forward * (1.0 - 0.2 * obs.veer_strength))
        
        return (lateral, forward, 0, yaw_cmd)

    def _compute_vertical_command(self, obs: ObstacleInfo, target_vert: int = 0) -> int:
        """
        V17: Compute vertical command with strict ceiling protection.
        """
        # If altitude locked, use altitude correction
        if self.altitude_locked:
            return self._compute_altitude_correction(obs)
        
        # Apply deadzone to reduce bobbing
        if abs(target_vert) < VERTICAL_DEADZONE * SPEED_VERTICAL:
            return 0
        
        # Ceiling protection - absolute limit
        if self.current_altitude >= MAX_ALTITUDE - 0.1:
            if target_vert > 0:
                return -SPEED_VERTICAL_SLOW  # Force descent
            return target_vert
        
        # Ceiling proximity - reduce up movement
        if self.current_altitude >= MAX_ALTITUDE - 0.3:
            if target_vert > 0:
                return 0  # Stop rising
        
        # Check ceiling from obstacle detection
        if target_vert > 0 and not obs.can_go_up:
            return 0
        
        # Floor protection
        if target_vert < 0 and not obs.can_go_down:
            return 0
        
        # Altitude limits
        if self.current_altitude <= MIN_ALTITUDE and target_vert < 0:
            return 0
        
        # Apply damping to reduce oscillation
        return int(target_vert * VERTICAL_DAMPING)

    def get_action(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[any], Tuple[int, int, int, int], np.ndarray, np.ndarray]:
        """
        V17: Main decision loop with adaptive speed and ceiling protection.
        """
        depth_vis, depth_meters = self.depth_engine.infer(frame)
        obs = self.obstacle_detector.analyze(depth_meters, self.current_altitude)
        self.update_voxels(depth_meters)
        
        self.seg_frame_count += 1
        if self.seg_frame_count >= SEG_INFERENCE_INTERVAL or self.last_floor_mask is None:
            self.seg_frame_count = 0
            self.last_floor_mask, self.last_seg_vis = self.floor_engine.segment_floor(frame)
        
        floor_mask = self.last_floor_mask if self.last_floor_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
        seg_vis = self.last_seg_vis if self.last_seg_vis is not None else frame.copy()
        
        self.update_floor_map(depth_meters, floor_mask)
        
        target = self._detect_target(frame)
        targets = target['boxes'] if target else None
        
        # V18: Track target detection for UI display
        if targets and len(targets) > 0:
            self.target_detected = True
            self.last_target_seen_time = time.time()
            self.target_detection_count += 1
        
        self.update_path()
        path_overlay = self.get_path_overlay(frame)
        
        # V17: Check if rotation is complete
        if self.escape_state != EscapeState.NONE and obs.center_dist > SAFE_DIST:
            self.escape_state = EscapeState.NONE
            self._end_rotation()
        
        # V17: Get adaptive speed for this frame
        adaptive_speed = obs.recommended_speed
        
        # ============= CORNER ESCAPE MODE =============
        if (obs.is_corner and self.obstacle_detector.is_stuck_in_corner()) or \
           self.escape_state in [EscapeState.ROTATING_LEFT, EscapeState.ROTATING_RIGHT]:
            self.mode = "🔄 CORNER ESCAPE"
            rc = self._handle_corner_escape(obs)
            return (depth_vis, targets, rc, seg_vis, path_overlay)
        
        # ============= CEILING WARNING =============
        if obs.ceiling_detected or self.current_altitude >= MAX_ALTITUDE - 0.2:
            vert_correction = -SPEED_VERTICAL_SLOW if self.current_altitude > MAX_ALTITUDE - 0.3 else 0
            if obs.is_emergency:
                self.mode = "🛑 CEILING+EMERGENCY"
                self.sub_mode = f"ALT:{self.current_altitude:.1f}m (MAX:{MAX_ALTITUDE}m)"
                self._start_rotation()  # Lock altitude
                return depth_vis, targets, (0, SPEED_VALUES[SpeedMode.STOP], vert_correction, SPEED_YAW_NORMAL), seg_vis, path_overlay
        
        # ============= EMERGENCY STOP =============
        if obs.is_emergency or self.obstacle_detector.is_persistent_emergency():
            self.mode = "🛑 EMERGENCY"
            self.sub_mode = f"STOP @ {obs.center_dist:.2f}m"
            self.consecutive_safe_frames = 0
            self._end_rotation()  # Not rotating
            
            vert = self._compute_vertical_command(obs, SPEED_VERTICAL_SLOW if obs.can_go_up else 0)
            yaw_cmd = obs.escape_rotation if obs.escape_rotation != 0 else SPEED_YAW_NORMAL
            
            return depth_vis, targets, (0, SPEED_VALUES[SpeedMode.STOP], vert, yaw_cmd), seg_vis, path_overlay
        
        # ============= CRITICAL - SLOW VEER =============
        if obs.is_critical:
            self.mode = "⚠️ CRITICAL"
            self.sub_mode = f"SLOW @ {obs.center_dist:.2f}m | {obs.speed_mode.value}"
            self.consecutive_safe_frames = 0
            self._end_rotation()
            
            # V17: Use adaptive speed (will be CREEP or SLOW)
            lat, fwd, _, yaw_cmd = self._compute_veering_command(obs, adaptive_speed)
            
            if yaw_cmd == 0:
                yaw_cmd = SPEED_YAW_SLOW if obs.right_dist > obs.left_dist else -SPEED_YAW_SLOW
            
            vert = self._compute_vertical_command(obs, 0)
            
            return depth_vis, targets, (lat, fwd, vert, yaw_cmd), seg_vis, path_overlay
        
        # ============= DANGER - CAREFUL VEER =============
        if obs.is_danger:
            self.mode = "⚡ DANGER"
            self.sub_mode = f"CAREFUL @ {obs.center_dist:.2f}m | {obs.speed_mode.value}"
            self.consecutive_safe_frames = 0
            self._end_rotation()
            
            if obs.thin_obstacle_detected:
                self.sub_mode = f"POLE @ col{obs.thin_obstacle_column}"
                lat = -SPEED_VEER_LATERAL if obs.thin_obstacle_column < THIN_OBSTACLE_COLUMNS // 2 else SPEED_VEER_LATERAL
                yaw_cmd = SPEED_YAW_SLOW if obs.thin_obstacle_column < THIN_OBSTACLE_COLUMNS // 2 else -SPEED_YAW_SLOW
                return depth_vis, targets, (lat, adaptive_speed, 0, yaw_cmd), seg_vis, path_overlay
            
            if target and target['size'] > 0.05:
                self.mode = "🎯 CAREFUL"
                h_err = target['cx'] - 0.5
                yaw_cmd = int(self.pid_yaw.compute(h_err * 100))
                return depth_vis, targets, (0, adaptive_speed, 0, yaw_cmd), seg_vis, path_overlay
            
            # V17: Veer with adaptive speed
            lat, fwd, _, yaw_cmd = self._compute_veering_command(obs, adaptive_speed)
            
            return depth_vis, targets, (lat, fwd, 0, yaw_cmd), seg_vis, path_overlay
        
        # ============= CAUTION - MODERATE SPEED =============
        if obs.is_caution:
            self.consecutive_safe_frames = 0
            self._end_rotation()
            
            if target:
                rc = self._handle_target(target, obs, speed_limit=adaptive_speed)
                return depth_vis, targets, rc, seg_vis, path_overlay
            
            self.mode = "👀 CAUTION"
            self.sub_mode = f"MODERATE @ {obs.center_dist:.2f}m | {obs.speed_mode.value}"
            
            # V17: Mild veering with adaptive speed
            lat, fwd, _, yaw_cmd = self._compute_veering_command(obs, adaptive_speed)
            
            return depth_vis, targets, (lat, fwd, 0, yaw_cmd), seg_vis, path_overlay
        
        # ============= SAFE/CLEAR - ADAPTIVE SPEED =============
        self.consecutive_safe_frames += 1
        self._end_rotation()
        
        if target:
            rc = self._handle_target(target, obs, speed_limit=adaptive_speed)
            return depth_vis, targets, rc, seg_vis, path_overlay
        
        # Stuck detection
        dist_moved = np.sqrt(sum((a - b)**2 for a, b in zip(
            (self.pos_x, self.pos_y, self.pos_z), self.last_pos)))
        
        if dist_moved < STUCK_THRESH:
            if time.time() - self.last_move_time > STUCK_TIME:
                self.mode = "🔄 UNSTUCK"
                self.sub_mode = "ROTATING"
                self._start_rotation()  # Lock altitude during unstuck rotation
                self.last_move_time = time.time()
                self.explore_direction *= -1
                vert_correction = self._compute_altitude_correction(obs)
                return depth_vis, targets, (0, 0, vert_correction, SPEED_YAW_NORMAL * self.explore_direction), seg_vis, path_overlay
        else:
            self.last_pos = (self.pos_x, self.pos_y, self.pos_z)
            self.last_move_time = time.time()
        
        # Explore with adaptive speed
        rc = self._handle_explore(obs)
        return depth_vis, targets, rc, seg_vis, path_overlay

    def _handle_target(self, target: dict, obs: ObstacleInfo, speed_limit: int) -> Tuple[int, int, int, int]:
        cx, cy, size = target['cx'], target['cy'], target['size']
        
        h_err = cx - 0.5
        v_err = cy - 0.5
        size_err = GOAL_SIZE_TARGET - size
        
        h_centered = abs(h_err) < GOAL_CENTER_TOLERANCE
        v_centered = abs(v_err) < GOAL_VERTICAL_TOLERANCE
        size_ok = abs(size_err) < 0.05
        
        yaw_cmd = int(self.pid_yaw.compute(h_err * 100))
        
        # V17: Reduced vertical command for less bobbing
        raw_vert = int(self.pid_vert.compute(-v_err * 50))  # Reduced from 60
        vert_cmd = self._compute_vertical_command(obs, raw_vert) if not v_centered else 0
        
        if h_centered and v_centered and size_ok:
            self.mode = "✅ LOCKED"
            self.sub_mode = "TARGET CENTERED"
            return (0, 0, 0, yaw_cmd)
        
        if h_centered and v_centered:
            self.mode = "🎯 APPROACH"
            
            if obs.center_dist < DANGER_DIST:
                self.sub_mode = "BLOCKED"
                return (0, 0, vert_cmd, yaw_cmd)
            
            # V17: Use adaptive speed for approach
            fwd = min(SPEED_APPROACH, speed_limit)
            self.sub_mode = f"FWD @ {fwd} | {obs.speed_mode.value}"
            return (0, fwd, vert_cmd, yaw_cmd)
        
        self.mode = "🔍 TRACKING"
        self.sub_mode = f"H:{h_err:.2f} V:{v_err:.2f}"
        
        fwd = 0
        if size < GOAL_SIZE_TARGET * 0.4 and obs.center_dist > CAUTION_DIST:
            fwd = min(obs.recommended_speed, speed_limit)
        
        return (0, fwd, vert_cmd, yaw_cmd)

    def _handle_explore(self, obs: ObstacleInfo) -> Tuple[int, int, int, int]:
        """V17: Explore with adaptive speed based on obstacle proximity."""
        self.mode = "🔭 EXPLORE"
        
        # V17: Use adaptive speed - faster when clear, slower near obstacles
        speed = obs.recommended_speed
        
        if obs.is_clear:
            self.sub_mode = f"FAST @ {obs.center_dist:.1f}m | {obs.speed_mode.value}"
        else:
            self.sub_mode = f"ADAPTIVE @ {obs.center_dist:.1f}m | {obs.speed_mode.value}"
        
        # Apply veering if needed even in explore mode
        if obs.veer_strength > 0.1:
            lat, fwd, _, yaw_cmd = self._compute_veering_command(obs, speed)
            return (lat, fwd, 0, yaw_cmd)
        
        yaw_cmd = int(obs.steer_amount * SPEED_YAW_SLOW * 0.4)
        
        return (0, speed, 0, yaw_cmd)


# ==========================================
# MANUAL CONTROLLER
# ==========================================
class ManualController:
    def get_rc(self, keys) -> Tuple[int, int, int, int]:
        lat, fwd, vert, yaw = 0, 0, 0, 0
        speed = 50
        
        if keys[pygame.K_w]: fwd = speed
        if keys[pygame.K_s]: fwd = -speed
        if keys[pygame.K_a]: lat = -speed
        if keys[pygame.K_d]: lat = speed
        if keys[pygame.K_UP]: vert = speed
        if keys[pygame.K_DOWN]: vert = -speed
        if keys[pygame.K_LEFT]: yaw = -speed
        if keys[pygame.K_RIGHT]: yaw = speed
        if keys[pygame.K_LSHIFT]:
            lat, fwd, vert, yaw = [int(v * 1.4) for v in [lat, fwd, vert, yaw]]
        
        return tuple(int(np.clip(v, -100, 100)) for v in [lat, fwd, vert, yaw])


# ==========================================
# V17 ENHANCED VISUALIZER
# ==========================================
class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1500, 850))
        pygame.display.set_caption("Tello V17.0 - Adaptive Speed + Advanced Pathfinding")
        self.font = pygame.font.Font(None, 26)
        self.font_sm = pygame.font.Font(None, 20)
        self.font_lg = pygame.font.Font(None, 32)
        self.clock = pygame.time.Clock()
    
    def render(self, frame, depth, brain, targets, auto, battery, rc, obs_info, 
               seg_vis=None, path_overlay=None, cam_fps=0):
        self.screen.fill((15, 15, 20))
        
        # ======================
        # LEFT COLUMN - Camera Views
        # ======================
        
        if path_overlay is not None:
            path_view = cv2.resize(path_overlay, (520, 390))
        else:
            path_view = cv2.resize(frame, (520, 390))
        
        if targets:
            for b in targets:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                sx, sy = 520/960, 390/720
                cv2.rectangle(path_view, (int(x1*sx), int(y1*sy)), 
                             (int(x2*sx), int(y2*sy)), (0, 255, 0), 2)
        
        surf = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(path_view, cv2.COLOR_BGR2RGB), (1, 0, 2)))
        self.screen.blit(surf, (10, 10))
        
        label_col = (0, 255, 255) if brain.floor_map.world_path else (180, 180, 180)
        self.screen.blit(self.font.render("Camera + Path (V17 B-Spline)", True, label_col), (10, 405))
        
        if seg_vis is not None:
            seg_small = cv2.resize(seg_vis, (255, 190))
            surf_seg = pygame.surfarray.make_surface(
                np.transpose(cv2.cvtColor(seg_small, cv2.COLOR_BGR2RGB), (1, 0, 2)))
            self.screen.blit(surf_seg, (10, 430))
            self.screen.blit(self.font_sm.render("Floor Segmentation", True, (180, 180, 180)), (10, 625))
        
        d_color = cv2.applyColorMap(cv2.resize(depth, (255, 190)), cv2.COLORMAP_MAGMA)
        
        if obs_info.is_emergency:
            cv2.rectangle(d_color, (0, 0), (254, 189), (0, 0, 255), 6)
        elif obs_info.ceiling_detected:
            cv2.rectangle(d_color, (0, 0), (254, 189), (255, 255, 0), 4)
        elif obs_info.is_corner:
            cv2.rectangle(d_color, (0, 0), (254, 189), (255, 0, 255), 4)
        elif obs_info.is_critical:
            cv2.rectangle(d_color, (0, 0), (254, 189), (0, 128, 255), 4)
        
        surf = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB), (1, 0, 2)))
        self.screen.blit(surf, (275, 430))
        self.screen.blit(self.font_sm.render(f"Depth: {obs_info.center_dist:.2f}m", True, (180, 180, 180)), (275, 625))
        
        # ======================
        # MIDDLE COLUMN - Maps
        # ======================
        
        flat = np.max(brain.voxels, axis=2)
        map_vis = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        map_vis[flat == 1] = (150, 150, 150)
        map_vis[brain.visited == 1] = (0, 50, 0)
        
        rx = np.clip(int(brain.pos_x / VOXEL_RES) + CENTER, 2, GRID_SIZE - 3)
        rz = np.clip(int(brain.pos_z / VOXEL_RES) + CENTER, 2, GRID_SIZE - 3)
        cv2.circle(map_vis, (rx, rz), 4, (0, 255, 0), -1)
        
        yaw_rad = np.radians(-brain.yaw)
        hx = int(rx + 12 * np.sin(yaw_rad))
        hz = int(rz - 12 * np.cos(yaw_rad))
        cv2.line(map_vis, (rx, rz), (hx, hz), (0, 255, 255), 2)
        
        surf = pygame.transform.scale(
            pygame.surfarray.make_surface(np.transpose(map_vis, (1, 0, 2))), (200, 200))
        self.screen.blit(surf, (550, 10))
        self.screen.blit(self.font_sm.render("Obstacle Map", True, (180, 180, 180)), (550, 215))
        
        floor_vis = brain.floor_map.get_visualization_2d()
        
        fx = np.clip(int(brain.pos_x / FLOOR_VOXEL_RES) + FLOOR_CENTER, 0, FLOOR_GRID_SIZE - 1)
        fz = np.clip(int(brain.pos_z / FLOOR_VOXEL_RES) + FLOOR_CENTER, 0, FLOOR_GRID_SIZE - 1)
        cv2.circle(floor_vis, (fx, fz), 3, (0, 255, 255), -1)
        
        fx_end = int(fx + 10 * np.sin(yaw_rad))
        fz_end = int(fz - 10 * np.cos(yaw_rad))
        cv2.line(floor_vis, (fx, fz), (fx_end, fz_end), (255, 255, 0), 2)
        
        floor_vis_small = cv2.resize(floor_vis, (200, 200), interpolation=cv2.INTER_NEAREST)
        surf_floor = pygame.surfarray.make_surface(np.transpose(floor_vis_small, (1, 0, 2)))
        self.screen.blit(surf_floor, (550, 240))
        self.screen.blit(self.font_sm.render("Floor Map + Smooth Path", True, (180, 180, 180)), (550, 445))
        
        # ======================
        # RIGHT COLUMN - Status
        # ======================
        col_x = 780
        y = 10
        
        self.screen.blit(self.font_lg.render("TELLO V17.1", True, (100, 200, 255)), (col_x, y))
        y += 35
        
        # V18: TARGET DETECTION STATUS PANEL
        pygame.draw.rect(self.screen, (40, 40, 50), (col_x - 5, y - 5, 220, 55))
        if brain.target_detected:
            time_since = time.time() - brain.last_target_seen_time
            if time_since < 2.0:
                target_col = (0, 255, 0)  # Green - just seen
                status = "TARGET VISIBLE"
            else:
                target_col = (255, 255, 0)  # Yellow - seen recently
                status = f"LAST SEEN {time_since:.1f}s"
            self.screen.blit(self.font.render(f"🎯 {status}", True, target_col), (col_x, y))
            y += 22
            self.screen.blit(self.font_sm.render(f"Detections: {brain.target_detection_count}", True, (150, 255, 150)), (col_x, y))
        else:
            self.screen.blit(self.font.render("🔍 SEARCHING...", True, (150, 150, 150)), (col_x, y))
            y += 22
            self.screen.blit(self.font_sm.render("No target detected yet", True, (100, 100, 100)), (col_x, y))
        y += 30
        
        mode_col = (0, 255, 0) if auto else (100, 100, 255)
        self.screen.blit(self.font.render(
            f"{'AUTO (ADAPTIVE)' if auto else 'MANUAL'}", True, mode_col), (col_x, y))
        y += 30
        
        state_col = (255, 0, 0) if "EMERGENCY" in brain.mode else \
                    (255, 255, 0) if "CEILING" in brain.mode else \
                    (255, 0, 255) if "CORNER" in brain.mode else \
                    (255, 200, 0) if "CRITICAL" in brain.mode or "DANGER" in brain.mode else \
                    (0, 255, 0) if "LOCKED" in brain.mode else (200, 200, 200)
        
        self.screen.blit(self.font.render(brain.mode, True, state_col), (col_x, y))
        y += 25
        self.screen.blit(self.font_sm.render(brain.sub_mode, True, (150, 150, 150)), (col_x, y))
        y += 30
        
        # V17: Speed mode indicator with color
        speed_colors = {
            SpeedMode.STOP: (255, 0, 0),
            SpeedMode.CREEP: (255, 100, 0),
            SpeedMode.SLOW: (255, 200, 0),
            SpeedMode.CRUISE: (100, 255, 100),
            SpeedMode.FAST: (0, 255, 0),
        }
        speed_col = speed_colors.get(obs_info.speed_mode, (180, 180, 180))
        self.screen.blit(self.font.render(
            f"SPEED: {obs_info.speed_mode.value} ({obs_info.recommended_speed})", True, speed_col), (col_x, y))
        y += 25
        
        # V17: Altitude lock indicator
        if brain.altitude_locked:
            self.screen.blit(self.font_sm.render(
                f"🔒 ALT LOCKED: {brain.target_altitude:.2f}m", True, (255, 255, 0)), (col_x, y))
            y += 20
        
        # V17: Ceiling indicator
        if obs_info.ceiling_detected:
            self.screen.blit(self.font_sm.render(
                f"⚠️ CEILING: {obs_info.ceiling_dist:.2f}m", True, (255, 255, 0)), (col_x, y))
            y += 20
        
        # Veering indicator
        if obs_info.veer_strength > 0.1:
            veer_text = f"VEER {obs_info.veer_direction.upper()}: {obs_info.veer_strength:.0%}"
            self.screen.blit(self.font_sm.render(veer_text, True, (0, 255, 255)), (col_x, y))
            y += 20
        
        if brain.escape_state != EscapeState.NONE:
            self.screen.blit(self.font_sm.render(
                f"Escape: {brain.escape_state.value}", True, (255, 100, 255)), (col_x, y))
            y += 20
        
        if obs_info.is_corner:
            self.screen.blit(self.font_sm.render(
                f"CORNER: {obs_info.corner_type}", True, (255, 0, 255)), (col_x, y))
            y += 20
        
        y += 10
        
        self.screen.blit(self.font_sm.render("─── Distances ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        self.screen.blit(self.font_sm.render(
            f"Center: {obs_info.center_dist:.2f}m", True, (180, 180, 180)), (col_x, y))
        y += 18
        self.screen.blit(self.font_sm.render(
            f"L: {obs_info.left_dist:.2f}m  R: {obs_info.right_dist:.2f}m", True, (180, 180, 180)), (col_x, y))
        y += 18
        self.screen.blit(self.font_sm.render(
            f"Top: {obs_info.top_dist:.2f}m  Bot: {obs_info.bottom_dist:.2f}m", True, (180, 180, 180)), (col_x, y))
        y += 30
        
        self.screen.blit(self.font_sm.render("─── Position ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        self.screen.blit(self.font_sm.render(
            f"X: {brain.pos_x:.2f}  Y: {brain.pos_y:.2f}  Z: {brain.pos_z:.2f}",
            True, (180, 180, 180)), (col_x, y))
        y += 18
        
        # V17: Altitude with limit indicator
        alt_col = (255, 255, 0) if brain.current_altitude > MAX_ALTITUDE - 0.3 else (180, 180, 180)
        self.screen.blit(self.font_sm.render(
            f"Altitude: {brain.current_altitude:.2f}m (max {MAX_ALTITUDE}m)", True, alt_col), (col_x, y))
        y += 18
        self.screen.blit(self.font_sm.render(
            f"Yaw: {brain.yaw:.0f}°", True, (180, 180, 180)), (col_x, y))
        y += 30
        
        self.screen.blit(self.font_sm.render("─── Pathfinding V17 ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        
        path_features = []
        if PATH_STRING_PULL_ENABLED:
            path_features.append("StringPull")
        if PATH_BSPLINE_ENABLED and SCIPY_AVAILABLE:
            path_features.append("B-Spline")
        path_text = " + ".join(path_features) if path_features else "A* Only"
        self.screen.blit(self.font_sm.render(f"Path: {path_text}", True, (0, 255, 0)), (col_x, y))
        y += 18
        
        path_len = len(brain.floor_map.world_path)
        if path_len > 0:
            self.screen.blit(self.font_sm.render(f"Waypoints: {path_len} (smoothed)", True, (0, 255, 255)), (col_x, y))
        else:
            self.screen.blit(self.font_sm.render("Path: Building...", True, (255, 255, 0)), (col_x, y))
        y += 30
        
        self.screen.blit(self.font_sm.render("─── Controls ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        fwd_color = (100, 150, 255) if rc[1] >= 0 else (255, 100, 100)
        self.screen.blit(self.font_sm.render(
            f"RC: L={rc[0]:+3d} F={rc[1]:+3d} V={rc[2]:+3d} Y={rc[3]:+3d}",
            True, fwd_color), (col_x, y))
        y += 30
        
        batt_col = (0, 255, 0) if battery > 30 else (255, 165, 0) if battery > 15 else (255, 0, 0)
        self.screen.blit(self.font.render(f"BATTERY: {battery}%", True, batt_col), (col_x, y))
        y += 25
        
        # V17: Video FPS indicator
        fps_col = (0, 255, 0) if cam_fps > 25 else (255, 255, 0) if cam_fps > 15 else (255, 0, 0)
        self.screen.blit(self.font_sm.render(f"Video FPS: {cam_fps:.1f}", True, fps_col), (col_x, y))
        y += 18
        
        model_text = "METRIC DEPTH ✓" if brain.depth_engine.is_metric else "RELATIVE DEPTH"
        model_col = (0, 255, 100) if brain.depth_engine.is_metric else (255, 150, 0)
        self.screen.blit(self.font_sm.render(model_text, True, model_col), (col_x, y))
        y += 18
        
        self.screen.blit(self.font_sm.render("OFFLINE MODE ✓ | LOW-LATENCY VIDEO", True, (0, 200, 100)), (col_x, y))
        y += 40
        
        # ======================
        # BOTTOM - Instructions
        # ======================
        y = 660
        instructions = [
            "[SPACE] Takeoff/Land  [M] Auto/Manual  [ESC] Quit",
            "[WASD] Move  [Arrows] Vert/Yaw  [SHIFT] Fast",
            f"Target: {brain.target_obj}",
            "[T] Set target at current direction",
            "V17: ADAPTIVE SPEED | CEILING PROTECTION | B-SPLINE PATHS",
            f"Speed zones: STOP<{EMERGENCY_STOP_DIST}m | SLOW<{DANGER_DIST}m | CRUISE<{SAFE_DIST}m | FAST>{CLEAR_DIST}m",
            f"Max altitude: {MAX_ALTITUDE}m | Altitude lock during rotation"
        ]
        for txt in instructions:
            self.screen.blit(self.font_sm.render(txt, True, (100, 100, 100)), (10, y))
            y += 18
        
        pygame.display.flip()
        return self.clock.tick(30)


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("  TELLO V17.0: ADAPTIVE SPEED + ADVANCED PATHFINDING")
    print("=" * 60)
    print("  V17 IMPROVEMENTS:")
    print("  - ADAPTIVE SPEED: Auto-adjusts based on obstacle distance")
    print(f"    * FAST when clear (>{CLEAR_DIST}m)")
    print(f"    * CRUISE when safe ({SAFE_DIST}-{CLEAR_DIST}m)")
    print(f"    * SLOW when danger (<{DANGER_DIST}m)")
    print(f"    * STOP in emergency (<{EMERGENCY_STOP_DIST}m)")
    print("  - CEILING PROTECTION: Strict altitude limits during rotation")
    print(f"    * Max altitude: {MAX_ALTITUDE}m")
    print("    * Altitude lock during all rotations")
    print("  - IMPROVED PATHFINDING:")
    print("    * A* search")
    print("    * String Pulling (removes unnecessary waypoints)")
    print("    * B-Spline smoothing (rounds corners)")
    print("  - LOW-LATENCY VIDEO: Optimized capture settings")
    print("")
    print("  CONTROLS:")
    print("  - [T] Set target at current direction")
    print("  - [M] Auto/Manual mode")
    print("=" * 60)
    
    target = input("Target object (default 'cup'): ").strip() or "cup"
    
    drone = Tello()
    try:
        drone.connect()
        drone.streamon()
        print(f"[DRONE] Connected! Battery: {drone.get_battery()}%")
        
        cam = ThreadedCam().start()
        time.sleep(2)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    brain = SafeVoxelBrain(target)
    manual = ManualController()
    viz = Visualizer()
    
    brain.start_visualization()
    
    running = True
    auto_mode = False
    rc_cmd = (0, 0, 0, 0)
    obs_info = ObstacleInfo()
    seg_vis = None
    path_overlay = None
    
    print("[READY] SPACE=takeoff  M=auto/manual  T=set target  ESC=quit")
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if drone.is_flying:
                            drone.land()
                        else:
                            drone.takeoff()
                            time.sleep(0.5)
                    if event.key == pygame.K_m:
                        auto_mode = not auto_mode
                        drone.send_rc_control(0, 0, 0, 0)
                        brain.pid_yaw.reset()
                        brain.pid_vert.reset()
                        brain.escape_state = EscapeState.NONE
                        brain._end_rotation()  # Release any altitude lock
                    if event.key == pygame.K_t:
                        target_x = brain.pos_x + DEFAULT_TARGET_DISTANCE * np.sin(np.radians(-brain.yaw))
                        target_z = brain.pos_z - DEFAULT_TARGET_DISTANCE * np.cos(np.radians(-brain.yaw))
                        brain.set_target_from_detection(target_x, target_z)
                        brain.default_target_set = True
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame = cv2.resize(frame, (960, 720))
            
            try:
                imu_yaw = float(drone.get_yaw())
                imu_height = float(drone.get_height()) / 100.0
            except:
                imu_yaw = brain.yaw
                imu_height = None
            
            brain.update_vo(frame, imu_yaw, imu_height)
            
            if auto_mode and drone.is_flying:
                depth_vis, targets, rc_cmd, seg_vis, path_overlay = brain.get_action(frame)
                
                # SAFETY CHECK: No backward flight in auto mode
                if rc_cmd[1] < 0:
                    print("[SAFETY] Blocked backward command in auto mode!")
                    rc_cmd = (rc_cmd[0], 0, rc_cmd[2], rc_cmd[3])
                
                rc_cmd = tuple(int(v) for v in rc_cmd)
                drone.send_rc_control(*rc_cmd)
                obs_info = brain.obstacle_detector.history[-1] if brain.obstacle_detector.history else ObstacleInfo()
            else:
                rc_cmd = manual.get_rc(pygame.key.get_pressed())
                if drone.is_flying:
                    drone.send_rc_control(*rc_cmd)
                depth_vis, depth_m = brain.depth_engine.infer(frame)
                obs_info = brain.obstacle_detector.analyze(depth_m, brain.current_altitude)
                
                if brain.floor_engine.enabled:
                    floor_mask, seg_vis = brain.floor_engine.segment_floor(frame)
                    brain.update_floor_map(depth_m, floor_mask)
                
                brain.update_path()
                path_overlay = brain.get_path_overlay(frame)
                
                targets = None
            
            try:
                battery = drone.get_battery()
            except:
                battery = 0
            
            cam_fps = cam.get_fps()
            
            viz.render(frame, depth_vis, brain, targets, auto_mode, battery, rc_cmd, 
                      obs_info, seg_vis, path_overlay, cam_fps)
    
    finally:
        print("[SHUTDOWN] Cleaning up...")
        brain.stop_visualization()
        drone.send_rc_control(0, 0, 0, 0)
        if drone.is_flying:
            drone.land()
        cam.stop()
        drone.streamoff()
        pygame.quit()
        print("[SHUTDOWN] Complete")


if __name__ == "__main__":
    main()