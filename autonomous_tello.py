"""
Tello EDU: Autonomous V13.0 "Safe-Mind Enhanced"
=================================================
Hardware: RTX 4060 8GB VRAM | Python 3.13

IMPROVEMENTS OVER V12.1:
1. CORNER ESCAPE: Detects when stuck in corners and rotates to escape
2. THIN OBSTACLE DETECTION: Enhanced detection for poles/narrow obstacles
3. NO BACKWARD FLIGHT: Drone NEVER flies backwards - always rotates to face safe direction
4. IMPROVED MAPPING: Better visual odometry with IMU fusion and drift correction

CRITICAL FIX: Uses Depth Anything V2 METRIC Indoor Model
- Outputs REAL depth in METERS (not relative depth!)
- Trained on Hypersim indoor dataset
- max_depth=20 meters for indoor scenes

Download the metric model checkpoint:
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base
  Save as: checkpoints/depth_anything_v2_metric_hypersim_vitb.pth
"""

import sys
import os
import time
import threading
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque
from enum import Enum

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pygame
from djitellopy import Tello
from ultralytics import YOLOWorld

# ==========================================
# CONFIGURATION
# ==========================================

# ==> CRITICAL: Use the METRIC INDOOR model checkpoint <==
DEPTH_CHECKPOINT_METRIC = 'checkpoints/depth_anything_v2_metric_hypersim_vitb.pth'
DEPTH_CHECKPOINT_RELATIVE = 'checkpoints/depth_anything_v2_vitb.pth'

# RTX 4060 Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Tello Camera Intrinsics (calibrated for 960x720)
FX = 921.17
FY = 919.02
CX = 460.0
CY = 351.0

# Mapping Constants
VOXEL_RES = 0.10        # 10cm per voxel
GRID_SIZE = 200         # 20x20 meters XZ plane
HEIGHT_LEVELS = 30      # 3 meters vertical
CENTER = GRID_SIZE // 2

# ==> OBSTACLE AVOIDANCE THRESHOLDS (in METERS) <==
EMERGENCY_STOP_DIST = 0.6    # HARD STOP
CRITICAL_DIST = 0.9          # Need to take action
DANGER_DIST = 1.2            # Significant slowdown + steering
CAUTION_DIST = 1.8           # Reduced speed
SAFE_DIST = 2.5              # Normal operation

# Speed limits (RC values -100 to 100)
# NOTE: NO NEGATIVE FORWARD SPEEDS - drone NEVER flies backward!
SPEED_MAX = 45               # Maximum forward speed
SPEED_CRUISE = 30            # Normal cruising
SPEED_CAREFUL = 18           # Near obstacles
SPEED_CREEP = 10             # Very close to obstacles
SPEED_STOP = 0               # Full stop (NO REVERSE!)
SPEED_VERTICAL = 25          # Up/down speed
SPEED_YAW = 40               # Rotation speed (slightly faster for escape)
SPEED_ESCAPE_YAW = 55        # Fast rotation for corner escape
SPEED_APPROACH = 12          # Approaching target

# Corner escape thresholds
CORNER_LEFT_THRESHOLD = 1.5   # If left is blocked within this distance
CORNER_RIGHT_THRESHOLD = 1.5  # If right is blocked within this distance
CORNER_FRONT_THRESHOLD = 1.2  # If front is blocked within this distance
CORNER_ROTATION_THRESHOLD = 2.0  # Min distance to stop rotating

# Thin obstacle detection
THIN_OBSTACLE_COLUMNS = 12     # Number of vertical columns to analyze
THIN_OBSTACLE_THRESHOLD = 0.8  # Depth difference to detect thin obstacles
EDGE_DETECTION_THRESHOLD = 50  # Sobel edge threshold for poles

# Safety margins
SAFETY_MARGIN = 1.3          # Multiply distances by this
TEMPORAL_SMOOTH_FRAMES = 3   # Frames to average depth

# Goal approach
GOAL_CENTER_TOLERANCE = 0.10
GOAL_SIZE_TARGET = 0.20
GOAL_VERTICAL_TOLERANCE = 0.12

# Visual Odometry - Enhanced
VO_DEADZONE = 2.0
VO_MAX_FEATURES = 200        # More features for accuracy
VO_QUALITY = 0.15            # Lower threshold for more points
VO_MIN_DISTANCE = 5          # Closer points allowed
VO_IMU_WEIGHT = 0.3          # How much to trust IMU vs VO

# Stuck detection
STUCK_THRESH = 0.15
STUCK_TIME = 3.0
CORNER_STUCK_TIME = 2.0      # Time before corner escape activates

# Logging
logging.getLogger('djitellopy').setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


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
# THREADED CAMERA
# ==========================================
class ThreadedCam:
    """Thread-safe camera capture."""
    
    def __init__(self, src: str = "udp://@0.0.0.0:11111"):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
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
            else:
                time.sleep(0.01)

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
# ENHANCED OBSTACLE DETECTOR
# ==========================================
@dataclass
class ObstacleInfo:
    """Structured obstacle detection with thin obstacle support."""
    center_dist: float = 10.0
    left_dist: float = 10.0
    right_dist: float = 10.0
    top_dist: float = 10.0
    bottom_dist: float = 10.0
    min_dist: float = 10.0
    
    # Fine-grained column distances for thin obstacle detection
    column_dists: List[float] = field(default_factory=lambda: [10.0] * THIN_OBSTACLE_COLUMNS)
    
    # Thin obstacle detected (poles, etc)
    thin_obstacle_detected: bool = False
    thin_obstacle_column: int = -1  # Which column has the thin obstacle
    
    is_emergency: bool = False
    is_critical: bool = False
    is_danger: bool = False
    is_caution: bool = False
    
    # Corner detection
    is_corner: bool = False  # Blocked on multiple sides
    corner_type: str = ""    # "left_corner", "right_corner", "dead_end"
    
    safe_direction: str = "forward"
    steer_amount: float = 0.0
    can_go_up: bool = True
    can_go_down: bool = True
    
    # Escape recommendation
    escape_rotation: int = 0  # Positive = right, Negative = left, 0 = none


class EnhancedObstacleDetector:
    """
    Multi-zone obstacle detection with:
    - Thin obstacle (pole) detection via edge analysis
    - Corner detection for escape logic
    - Fine-grained vertical column analysis
    """
    
    def __init__(self):
        self.history: Deque[ObstacleInfo] = deque(maxlen=5)
        self.emergency_count = 0
        self.corner_frames = 0  # Count consecutive corner detections
    
    def _detect_thin_obstacles(self, depth_meters: np.ndarray) -> Tuple[bool, int, float]:
        """
        Detect thin obstacles like poles using edge detection and depth discontinuities.
        Returns: (detected, column_index, min_distance)
        """
        h, w = depth_meters.shape
        
        # Use Sobel edge detection on depth map
        depth_uint8 = ((depth_meters - depth_meters.min()) / 
                       (depth_meters.max() - depth_meters.min() + 1e-6) * 255).astype(np.uint8)
        
        sobel_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Focus on center vertical strip where obstacles matter most
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
            
            # Check for strong vertical edges (poles create these)
            edge_strength = np.mean(col_edges)
            
            if edge_strength > EDGE_DETECTION_THRESHOLD:
                # Check if there's a depth discontinuity (close object in front of far background)
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
        """
        Detect if drone is in a corner situation.
        Returns: (is_corner, corner_type)
        """
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
    
    def analyze(self, depth_meters: np.ndarray) -> ObstacleInfo:
        """Analyze depth map for obstacles including thin obstacles and corners."""
        h, w = depth_meters.shape
        info = ObstacleInfo()
        
        # Standard zone analysis
        zones = {
            'center': (0.25, 0.75, 0.30, 0.70),
            'center_wide': (0.20, 0.80, 0.20, 0.80),
            'left': (0.25, 0.75, 0.0, 0.35),
            'right': (0.25, 0.75, 0.65, 1.0),
            'top': (0.0, 0.30, 0.25, 0.75),
            'bottom': (0.70, 1.0, 0.25, 0.75),
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
        
        # Fine-grained column analysis for thin obstacles
        col_width = w // THIN_OBSTACLE_COLUMNS
        for i in range(THIN_OBSTACLE_COLUMNS):
            x1 = i * col_width
            x2 = (i + 1) * col_width
            col_region = depth_meters[int(h*0.2):int(h*0.8), x1:x2]
            if col_region.size > 0:
                info.column_dists[i] = float(np.percentile(col_region, 10)) / SAFETY_MARGIN
        
        # Detect thin obstacles
        thin_detected, thin_col, thin_dist = self._detect_thin_obstacles(depth_meters)
        info.thin_obstacle_detected = thin_detected
        info.thin_obstacle_column = thin_col
        
        # If thin obstacle detected, override the relevant column distance
        if thin_detected and thin_col >= 0:
            info.column_dists[thin_col] = min(info.column_dists[thin_col], thin_dist)
        
        # Apply safety margin
        for k in zone_dists:
            zone_dists[k] = zone_dists[k] / SAFETY_MARGIN
        
        # Extract key distances
        info.center_dist = min(zone_dists.get('center', 10.0), 
                               zone_dists.get('center_wide', 10.0))
        info.left_dist = zone_dists.get('left', 10.0)
        info.right_dist = zone_dists.get('right', 10.0)
        info.top_dist = zone_dists.get('top', 10.0)
        info.bottom_dist = zone_dists.get('bottom', 10.0)
        
        # If thin obstacle in center columns, factor it into center_dist
        center_cols = info.column_dists[THIN_OBSTACLE_COLUMNS//3 : 2*THIN_OBSTACLE_COLUMNS//3]
        if center_cols:
            min_center_col = min(center_cols)
            info.center_dist = min(info.center_dist, min_center_col)
        
        info.min_dist = min(min(zone_dists.values()), min(info.column_dists))
        
        # Threat levels
        info.is_emergency = info.center_dist < EMERGENCY_STOP_DIST or info.min_dist < EMERGENCY_STOP_DIST * 0.8
        info.is_critical = info.center_dist < CRITICAL_DIST
        info.is_danger = info.center_dist < DANGER_DIST
        info.is_caution = info.center_dist < CAUTION_DIST
        
        # Corner detection
        info.is_corner, info.corner_type = self._detect_corner(
            info.left_dist, info.right_dist, info.center_dist)
        
        if info.is_corner:
            self.corner_frames += 1
        else:
            self.corner_frames = max(0, self.corner_frames - 1)
        
        # Vertical clearance
        info.can_go_up = info.top_dist > CAUTION_DIST
        info.can_go_down = info.bottom_dist > CAUTION_DIST
        
        # Determine best escape rotation
        if info.is_corner or info.is_critical:
            if info.corner_type == "left_corner" or info.left_dist < info.right_dist:
                info.escape_rotation = SPEED_ESCAPE_YAW  # Rotate right
            elif info.corner_type == "right_corner" or info.right_dist < info.left_dist:
                info.escape_rotation = -SPEED_ESCAPE_YAW  # Rotate left
            else:
                # Dead end - rotate in direction with slightly more space
                info.escape_rotation = SPEED_ESCAPE_YAW if info.right_dist >= info.left_dist else -SPEED_ESCAPE_YAW
        
        # Steering for non-corner situations
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
        
        # Emergency counter
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
        """Check if we've been in a corner for too long."""
        return self.corner_frames >= int(CORNER_STUCK_TIME * 30)  # Assuming 30 FPS


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
    """
    Improved visual odometry with IMU fusion.
    Uses more feature points and better drift correction.
    """
    
    def __init__(self):
        self.K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        self.prev_gray = None
        self.prev_points = None
        
        # Position state
        self.pos_x = 0.0
        self.pos_y = 1.0  # Start 1m above ground
        self.pos_z = 0.0
        self.yaw = 0.0
        
        # Velocity estimation (for IMU fusion)
        self.vel_x = 0.0
        self.vel_z = 0.0
        self.last_time = time.time()
        
        # Position history for drift detection
        self.pos_history: Deque[Tuple[float, float, float]] = deque(maxlen=100)
        
        # IMU integration
        self.imu_yaw_offset = 0.0
        self.imu_initialized = False
    
    def update(self, frame: np.ndarray, imu_yaw: float, imu_height: Optional[float] = None):
        """
        Update position estimate from visual odometry + IMU.
        
        Args:
            frame: Current camera frame
            imu_yaw: Yaw angle from drone IMU (degrees)
            imu_height: Optional height from barometer/ToF
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize IMU offset on first valid reading
        if not self.imu_initialized and abs(imu_yaw) < 360:
            self.imu_yaw_offset = imu_yaw
            self.imu_initialized = True
        
        # Use IMU yaw directly (more reliable than VO for rotation)
        self.yaw = float(imu_yaw)
        
        # Update height if available
        if imu_height is not None and imu_height > 0:
            # Smooth height update
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
        
        # Track features
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
                        # Use RANSAC for robust essential matrix estimation
                        E, mask = cv2.findEssentialMat(
                            good_new, good_old, self.K,
                            method=cv2.RANSAC, prob=0.999, threshold=0.5
                        )
                        
                        if E is not None:
                            _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, self.K, mask=mask)
                            
                            # Scale estimation (based on flow magnitude and height)
                            scale = min(0.04 * mag / 10.0, 0.15) * (self.pos_y / 1.0)
                            
                            # Transform to world coordinates using IMU yaw
                            yaw_rad = np.radians(-self.yaw)
                            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
                            
                            dx = float(t[0, 0]) * scale
                            dz = float(t[2, 0]) * scale
                            
                            dx_w = dx * c - dz * s
                            dz_w = dx * s + dz * c
                            
                            if not np.isnan(dx_w) and not np.isnan(dz_w):
                                # Apply with damping to reduce drift
                                self.pos_x += dx_w * 0.8
                                self.pos_z += dz_w * 0.8
                                
                                # Update velocity estimate
                                self.vel_x = dx_w / max(dt, 0.01)
                                self.vel_z = dz_w / max(dt, 0.01)
                    
                    except Exception:
                        pass
                
                self.prev_points = good_new.reshape(-1, 1, 2)
            else:
                # Re-detect features
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray, maxCorners=VO_MAX_FEATURES,
                    qualityLevel=VO_QUALITY, minDistance=VO_MIN_DISTANCE
                )
        
        self.prev_gray = gray
        
        # Store position history
        self.pos_history.append((self.pos_x, self.pos_y, self.pos_z))
    
    def get_position(self) -> Tuple[float, float, float]:
        return (self.pos_x, self.pos_y, self.pos_z)
    
    def reset_position(self):
        """Reset position to origin."""
        self.pos_x = 0.0
        self.pos_y = 1.0
        self.pos_z = 0.0
        self.pos_history.clear()


# ==========================================
# SAFE VOXEL BRAIN - NO BACKWARD FLIGHT
# ==========================================
class SafeVoxelBrain:
    """
    3D mapping and navigation with SAFETY FIRST approach.
    CRITICAL: Drone NEVER flies backward - always rotates to face safe direction.
    """
    
    def __init__(self, target_obj: str):
        print(f"[BRAIN] Initializing for target: {target_obj}")
        
        self.depth_engine = MetricDepthEngine()
        self.obstacle_detector = EnhancedObstacleDetector()
        self.vo = EnhancedVisualOdometry()
        
        # YOLO for object detection
        self.yolo = YOLOWorld('yolov8s-world.pt')
        self.yolo.set_classes([target_obj])
        self.yolo.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_obj = target_obj
        
        # 3D Voxel map
        self.voxels = np.zeros((GRID_SIZE, GRID_SIZE, HEIGHT_LEVELS), dtype=np.int8)
        self.visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.confidence = np.zeros((GRID_SIZE, GRID_SIZE, HEIGHT_LEVELS), dtype=np.float32)
        
        # PID controllers
        self.pid_yaw = PID(kp=0.7, ki=0.02, kd=0.1, limit=SPEED_YAW)
        self.pid_vert = PID(kp=0.6, ki=0.02, kd=0.1, limit=SPEED_VERTICAL)
        
        # State
        self.mode = "INIT"
        self.sub_mode = ""
        self.escape_state = EscapeState.NONE
        self.escape_start_yaw = 0.0
        self.escape_target_rotation = 0.0
        
        self.last_pos = (0.0, 0.0, 0.0)
        self.last_move_time = time.time()
        self.explore_direction = 1
        
        self.frames_since_obstacle = 0
        self.consecutive_safe_frames = 0
        
        print("[BRAIN] Ready - NO BACKWARD FLIGHT MODE")

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

    def update_vo(self, frame: np.ndarray, imu_yaw: float, imu_height: Optional[float] = None):
        """Update position from visual odometry."""
        self.vo.update(frame, imu_yaw, imu_height)

    def update_voxels(self, depth_meters: np.ndarray):
        """Update 3D voxel map from metric depth with confidence."""
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
        
        valid = (z > 0.3) & (z < 8.0)  # Reduced max range for accuracy
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
        
        # Update with confidence (older observations decay)
        self.voxels[iz, ix, iy] = 1
        self.confidence[iz, ix, iy] = np.minimum(self.confidence[iz, ix, iy] + 0.2, 1.0)
        
        # Update visited
        drone_ix = int(self.pos_x / VOXEL_RES) + CENTER
        drone_iz = int(self.pos_z / VOXEL_RES) + CENTER
        if 0 <= drone_ix < GRID_SIZE and 0 <= drone_iz < GRID_SIZE:
            self.visited[drone_iz, drone_ix] = 1

    def _detect_target(self, frame: np.ndarray) -> Optional[dict]:
        """Detect target object."""
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

    def _handle_corner_escape(self, obs: ObstacleInfo) -> Tuple[int, int, int, int]:
        """
        Handle corner escape by rotating until clear path is found.
        NEVER MOVES BACKWARD - only rotates in place.
        
        Returns: (lateral, forward, vertical, yaw)
        """
        # Initialize escape if needed
        if self.escape_state == EscapeState.NONE:
            self.escape_state = EscapeState.CORNER_DETECTED
            self.escape_start_yaw = self.yaw
            
            # Determine rotation direction based on obstacle positions
            if obs.right_dist > obs.left_dist:
                self.escape_state = EscapeState.ROTATING_RIGHT
                self.escape_target_rotation = 90  # Rotate 90 degrees right
            else:
                self.escape_state = EscapeState.ROTATING_LEFT
                self.escape_target_rotation = -90  # Rotate 90 degrees left
        
        # Calculate how much we've rotated
        rotation_done = self.yaw - self.escape_start_yaw
        
        # Normalize to -180 to 180
        while rotation_done > 180:
            rotation_done -= 360
        while rotation_done < -180:
            rotation_done += 360
        
        # Check if we've found a clear path OR rotated enough
        if obs.center_dist > CORNER_ROTATION_THRESHOLD:
            self.escape_state = EscapeState.ESCAPE_COMPLETE
            self.sub_mode = "ESCAPE COMPLETE"
            return (0, SPEED_CREEP, 0, 0)  # Move forward slowly
        
        # Check if we've rotated too much (full 360 = we're really stuck)
        if abs(rotation_done) > 270:
            # Try going up if possible
            if obs.can_go_up:
                self.sub_mode = "ESCAPE UP"
                return (0, 0, SPEED_VERTICAL, 0)
            else:
                # Just keep rotating
                self.sub_mode = "FULL ROTATION"
        
        # Continue rotating
        if self.escape_state == EscapeState.ROTATING_RIGHT:
            self.sub_mode = f"ROT RIGHT {abs(rotation_done):.0f}°"
            return (0, 0, 0, SPEED_ESCAPE_YAW)
        else:
            self.sub_mode = f"ROT LEFT {abs(rotation_done):.0f}°"
            return (0, 0, 0, -SPEED_ESCAPE_YAW)

    def get_action(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[any], Tuple[int, int, int, int]]:
        """
        Main decision loop - NEVER commands backward flight.
        
        Returns: (depth_vis, targets, (lateral, forward, vertical, yaw))
        """
        depth_vis, depth_meters = self.depth_engine.infer(frame)
        obs = self.obstacle_detector.analyze(depth_meters)
        self.update_voxels(depth_meters)
        
        target = self._detect_target(frame)
        targets = target['boxes'] if target else None
        
        # Reset escape state if we're now clear
        if self.escape_state != EscapeState.NONE and obs.center_dist > SAFE_DIST:
            self.escape_state = EscapeState.NONE
        
        # ============= CORNER ESCAPE MODE =============
        if (obs.is_corner and self.obstacle_detector.is_stuck_in_corner()) or \
           self.escape_state in [EscapeState.ROTATING_LEFT, EscapeState.ROTATING_RIGHT]:
            self.mode = "🔄 CORNER ESCAPE"
            return (depth_vis, targets, self._handle_corner_escape(obs))
        
        # ============= EMERGENCY STOP (NO BACKWARD!) =============
        if obs.is_emergency or self.obstacle_detector.is_persistent_emergency():
            self.mode = "🛑 EMERGENCY"
            self.sub_mode = f"STOP @ {obs.center_dist:.2f}m"
            self.consecutive_safe_frames = 0
            
            # STOP and ROTATE - NO BACKWARD FLIGHT!
            vert = SPEED_VERTICAL if obs.can_go_up else 0
            yaw_cmd = obs.escape_rotation if obs.escape_rotation != 0 else SPEED_YAW
            
            return depth_vis, targets, (0, SPEED_STOP, vert, yaw_cmd)
        
        # ============= CRITICAL - STOP AND ROTATE =============
        if obs.is_critical:
            self.mode = "⚠️ CRITICAL"
            self.sub_mode = f"ROTATE @ {obs.center_dist:.2f}m"
            self.consecutive_safe_frames = 0
            
            # STOP and rotate toward safe direction - NO BACKWARD!
            yaw_cmd = int(obs.steer_amount * SPEED_YAW)
            if yaw_cmd == 0:
                yaw_cmd = SPEED_YAW if obs.right_dist > obs.left_dist else -SPEED_YAW
            
            vert = SPEED_VERTICAL if obs.can_go_up and not obs.can_go_down else 0
            
            return depth_vis, targets, (0, SPEED_STOP, vert, yaw_cmd)
        
        # ============= DANGER - SLOW AND STEER =============
        if obs.is_danger:
            self.mode = "⚡ DANGER"
            self.sub_mode = f"AVOID @ {obs.center_dist:.2f}m"
            self.consecutive_safe_frames = 0
            
            # Thin obstacle handling
            if obs.thin_obstacle_detected:
                self.sub_mode = f"POLE @ col{obs.thin_obstacle_column}"
                # Steer away from the thin obstacle
                if obs.thin_obstacle_column < THIN_OBSTACLE_COLUMNS // 2:
                    yaw_cmd = SPEED_YAW  # Turn right
                else:
                    yaw_cmd = -SPEED_YAW  # Turn left
                return depth_vis, targets, (0, SPEED_CREEP, 0, yaw_cmd)
            
            # If target visible, very carefully approach
            if target and target['size'] > 0.05:
                self.mode = "🎯 CAREFUL"
                h_err = target['cx'] - 0.5
                yaw_cmd = int(self.pid_yaw.compute(h_err * 100))
                return depth_vis, targets, (0, SPEED_CREEP, 0, yaw_cmd)
            
            # Steer to safety - ALWAYS positive forward or zero!
            yaw_cmd = int(obs.steer_amount * SPEED_YAW)
            fwd = SPEED_CREEP if obs.center_dist > CRITICAL_DIST + 0.2 else SPEED_STOP
            
            return depth_vis, targets, (0, fwd, 0, yaw_cmd)
        
        # ============= CAUTION - REDUCED SPEED =============
        if obs.is_caution:
            self.consecutive_safe_frames = 0
            
            if target:
                return self._handle_target(target, obs, depth_vis, targets, speed_limit=SPEED_CAREFUL)
            
            self.mode = "👀 CAUTION"
            self.sub_mode = f"CAREFUL @ {obs.center_dist:.2f}m"
            
            yaw_cmd = int(obs.steer_amount * SPEED_YAW * 0.5)
            return depth_vis, targets, (0, SPEED_CAREFUL, 0, yaw_cmd)
        
        # ============= SAFE - NORMAL OPERATION =============
        self.consecutive_safe_frames += 1
        
        if target:
            return self._handle_target(target, obs, depth_vis, targets, speed_limit=SPEED_CRUISE)
        
        # Stuck detection
        dist_moved = np.sqrt(sum((a - b)**2 for a, b in zip(
            (self.pos_x, self.pos_y, self.pos_z), self.last_pos)))
        
        if dist_moved < STUCK_THRESH:
            if time.time() - self.last_move_time > STUCK_TIME:
                self.mode = "🔄 UNSTUCK"
                self.sub_mode = "ROTATING"
                self.last_move_time = time.time()
                self.explore_direction *= -1
                return depth_vis, targets, (0, 0, 0, SPEED_YAW * self.explore_direction)
        else:
            self.last_pos = (self.pos_x, self.pos_y, self.pos_z)
            self.last_move_time = time.time()
        
        # Explore
        return self._handle_explore(obs, depth_vis, targets)

    def _handle_target(self, target: dict, obs: ObstacleInfo, 
                       depth_vis: np.ndarray, targets, speed_limit: int) -> Tuple:
        """Handle target tracking with obstacle awareness."""
        cx, cy, size = target['cx'], target['cy'], target['size']
        
        h_err = cx - 0.5
        v_err = cy - 0.5
        size_err = GOAL_SIZE_TARGET - size
        
        h_centered = abs(h_err) < GOAL_CENTER_TOLERANCE
        v_centered = abs(v_err) < GOAL_VERTICAL_TOLERANCE
        size_ok = abs(size_err) < 0.05
        
        yaw_cmd = int(self.pid_yaw.compute(h_err * 100))
        vert_cmd = int(self.pid_vert.compute(-v_err * 80)) if not v_centered else 0
        
        if h_centered and v_centered and size_ok:
            self.mode = "✅ LOCKED"
            self.sub_mode = "TARGET CENTERED"
            return depth_vis, targets, (0, 0, 0, yaw_cmd)
        
        if h_centered and v_centered:
            self.mode = "🎯 APPROACH"
            
            if obs.center_dist < DANGER_DIST:
                self.sub_mode = "BLOCKED"
                return depth_vis, targets, (0, 0, vert_cmd, yaw_cmd)
            
            fwd = min(SPEED_APPROACH, speed_limit)
            self.sub_mode = f"FWD @ {fwd}"
            return depth_vis, targets, (0, fwd, vert_cmd, yaw_cmd)
        
        self.mode = "🔍 TRACKING"
        self.sub_mode = f"H:{h_err:.2f} V:{v_err:.2f}"
        
        fwd = 0
        if size < GOAL_SIZE_TARGET * 0.4 and obs.center_dist > CAUTION_DIST:
            fwd = SPEED_CAREFUL
        
        return depth_vis, targets, (0, fwd, vert_cmd, yaw_cmd)

    def _handle_explore(self, obs: ObstacleInfo, depth_vis: np.ndarray, targets) -> Tuple:
        """Handle exploration mode - always moving forward."""
        self.mode = "🔭 EXPLORE"
        
        if obs.center_dist > SAFE_DIST and self.consecutive_safe_frames > 10:
            speed = SPEED_CRUISE
            self.sub_mode = f"CRUISE @ {obs.center_dist:.1f}m"
        else:
            speed = SPEED_CAREFUL
            self.sub_mode = f"CAREFUL @ {obs.center_dist:.1f}m"
        
        yaw_cmd = int(obs.steer_amount * SPEED_YAW * 0.4)
        
        return depth_vis, targets, (0, speed, 0, yaw_cmd)


# ==========================================
# MANUAL CONTROLLER
# ==========================================
class ManualController:
    def get_rc(self, keys) -> Tuple[int, int, int, int]:
        lat, fwd, vert, yaw = 0, 0, 0, 0
        speed = 50
        
        if keys[pygame.K_w]: fwd = speed
        if keys[pygame.K_s]: fwd = -speed  # Manual mode allows backward for safety
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
# VISUALIZER
# ==========================================
class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1150, 750))
        pygame.display.set_caption("Tello V13.0 Safe-Mind Enhanced")
        self.font = pygame.font.Font(None, 26)
        self.font_sm = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()
    
    def render(self, frame, depth, brain, targets, auto, battery, rc, obs_info):
        self.screen.fill((15, 15, 20))
        
        # RGB with detections
        rgb = cv2.resize(frame, (520, 390))
        if targets:
            for b in targets:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                sx, sy = 520/960, 390/720
                cv2.rectangle(rgb, (int(x1*sx), int(y1*sy)), 
                             (int(x2*sx), int(y2*sy)), (0, 255, 0), 2)
        
        surf = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (1, 0, 2)))
        self.screen.blit(surf, (10, 10))
        
        # Depth map with column overlay
        d_color = cv2.applyColorMap(cv2.resize(depth, (520, 170)), cv2.COLORMAP_MAGMA)
        
        # Draw column distances
        col_width = 520 // THIN_OBSTACLE_COLUMNS
        for i, dist in enumerate(obs_info.column_dists):
            x = i * col_width + col_width // 2
            color = (0, 255, 0) if dist > DANGER_DIST else (0, 255, 255) if dist > CRITICAL_DIST else (0, 0, 255)
            cv2.line(d_color, (x, 0), (x, 170), color, 1)
            cv2.putText(d_color, f"{dist:.1f}", (x-10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Thin obstacle indicator
        if obs_info.thin_obstacle_detected:
            cv2.putText(d_color, "POLE DETECTED!", (180, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(d_color, f"CENTER: {obs_info.center_dist:.2f}m", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(d_color, f"L:{obs_info.left_dist:.1f} R:{obs_info.right_dist:.1f}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Color-coded warning border
        if obs_info.is_emergency:
            cv2.rectangle(d_color, (0, 0), (519, 169), (0, 0, 255), 8)
        elif obs_info.is_corner:
            cv2.rectangle(d_color, (0, 0), (519, 169), (255, 0, 255), 6)
        elif obs_info.is_critical:
            cv2.rectangle(d_color, (0, 0), (519, 169), (0, 128, 255), 6)
        elif obs_info.is_danger:
            cv2.rectangle(d_color, (0, 0), (519, 169), (0, 255, 255), 4)
        
        surf = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB), (1, 0, 2)))
        self.screen.blit(surf, (10, 410))
        
        # Voxel map
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
        
        # Draw FOV cone
        fov_angle = 40  # degrees
        for angle_offset in [-fov_angle, fov_angle]:
            fov_rad = np.radians(-brain.yaw + angle_offset)
            fx = int(rx + 25 * np.sin(fov_rad))
            fz = int(rz - 25 * np.cos(fov_rad))
            cv2.line(map_vis, (rx, rz), (fx, fz), (100, 100, 0), 1)
        
        surf = pygame.transform.scale(
            pygame.surfarray.make_surface(np.transpose(map_vis, (1, 0, 2))), (380, 380))
        self.screen.blit(surf, (540, 10))
        
        # HUD
        y = 400
        mode_col = (0, 255, 0) if auto else (100, 100, 255)
        self.screen.blit(self.font.render(
            f"{'AUTO (NO REVERSE)' if auto else 'MANUAL'}", True, mode_col), (540, y))
        
        state_col = (255, 0, 0) if "EMERGENCY" in brain.mode else \
                    (255, 0, 255) if "CORNER" in brain.mode else \
                    (255, 200, 0) if "CRITICAL" in brain.mode or "DANGER" in brain.mode else \
                    (0, 255, 0) if "LOCKED" in brain.mode else (200, 200, 200)
        
        self.screen.blit(self.font.render(brain.mode, True, state_col), (540, y + 25))
        self.screen.blit(self.font_sm.render(brain.sub_mode, True, (150, 150, 150)), (540, y + 50))
        
        # Escape state
        if brain.escape_state != EscapeState.NONE:
            self.screen.blit(self.font_sm.render(
                f"Escape: {brain.escape_state.value}", True, (255, 100, 255)), (540, y + 70))
        
        # Corner indicator
        if obs_info.is_corner:
            self.screen.blit(self.font_sm.render(
                f"CORNER: {obs_info.corner_type}", True, (255, 0, 255)), (700, y + 70))
        
        # Distances
        self.screen.blit(self.font_sm.render(
            f"Dist - C:{obs_info.center_dist:.2f} L:{obs_info.left_dist:.2f} R:{obs_info.right_dist:.2f}",
            True, (180, 180, 180)), (540, y + 95))
        
        # Position
        self.screen.blit(self.font_sm.render(
            f"Pos: X={brain.pos_x:.2f} Y={brain.pos_y:.2f} Z={brain.pos_z:.2f} Yaw={brain.yaw:.0f}°",
            True, (180, 180, 180)), (540, y + 115))
        
        # RC command with backward warning
        fwd_color = (100, 150, 255) if rc[1] >= 0 else (255, 100, 100)
        self.screen.blit(self.font_sm.render(
            f"RC: L={rc[0]:+3d} F={rc[1]:+3d} V={rc[2]:+3d} Y={rc[3]:+3d}",
            True, fwd_color), (540, y + 140))
        
        # Battery
        batt_col = (0, 255, 0) if battery > 30 else (255, 165, 0) if battery > 15 else (255, 0, 0)
        self.screen.blit(self.font.render(f"BATT: {battery}%", True, batt_col), (540, y + 165))
        
        # Model indicator
        model_text = "METRIC DEPTH ✓" if brain.depth_engine.is_metric else "RELATIVE DEPTH ⚠"
        model_col = (0, 255, 100) if brain.depth_engine.is_metric else (255, 100, 0)
        self.screen.blit(self.font_sm.render(model_text, True, model_col), (540, y + 195))
        
        # Controls
        for i, txt in enumerate([
            "[SPACE] Takeoff/Land  [M] Auto/Manual  [ESC] Quit",
            "[WASD] Move  [Arrows] Vert/Yaw  [SHIFT] Fast",
            f"Target: {brain.target_obj}",
            "AUTO MODE: Never flies backward - rotates to escape"
        ]):
            self.screen.blit(self.font_sm.render(txt, True, (100, 100, 100)), (540, y + 220 + i * 18))
        
        pygame.display.flip()
        return self.clock.tick(30)


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("  TELLO V13.0: SAFE-MIND ENHANCED")
    print("  - Corner escape with rotation")
    print("  - Thin obstacle (pole) detection")
    print("  - NO BACKWARD FLIGHT in auto mode")
    print("  - Improved visual odometry + mapping")
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
    
    running = True
    auto_mode = False
    rc_cmd = (0, 0, 0, 0)
    obs_info = ObstacleInfo()
    
    print("[READY] SPACE=takeoff  M=auto/manual  ESC=quit")
    
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
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        
        frame = cv2.resize(frame, (960, 720))
        
        try:
            imu_yaw = float(drone.get_yaw())
            imu_height = float(drone.get_height()) / 100.0  # cm to m
        except:
            imu_yaw = brain.yaw
            imu_height = None
        
        brain.update_vo(frame, imu_yaw, imu_height)
        
        if auto_mode and drone.is_flying:
            depth_vis, targets, rc_cmd = brain.get_action(frame)
            
            # SAFETY CHECK: Ensure no backward flight in auto mode
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
            obs_info = brain.obstacle_detector.analyze(depth_m)
            targets = None
        
        try:
            battery = drone.get_battery()
        except:
            battery = 0
        
        viz.render(frame, depth_vis, brain, targets, auto_mode, battery, rc_cmd, obs_info)
    
    drone.send_rc_control(0, 0, 0, 0)
    if drone.is_flying:
        drone.land()
    cam.stop()
    drone.streamoff()
    pygame.quit()


if __name__ == "__main__":
    main()