"""
Tello EDU: Autonomous V18.1 "Person-to-Goal Path Projection"
===================================================================
Hardware: RTX 4060 8GB VRAM | Python 3.13

NEW FEATURES IN V18.1:
1. PERSON-TO-GOAL PATH: Projects A* path from detected PERSON to GOAL
2. PERSON DETECTION: Always-on YOLO World person detection
3. PATH VISUALIZATION TOGGLE: Press T to toggle path overlay on/off
4. PATHFINDING STATUS LOGGING: Detailed status in visualization window
5. 3D POSITION ESTIMATION: Estimates person's world position from depth
6. WORKS WITHOUT GOAL IN VIEW: Projects path onto ground plane

INHERITED FROM V18.0:
- COLUMN DEPTH VISUALIZATION: 12-column depth analysis with visual overlay
- DECISIVE OBSTACLE AVOIDANCE: Actual rotation when obstacles detected
- SMART EXPLORATION: Prioritize furthest depth, avoid re-exploring
- EMERGENCY STOP: Absolute stop when <0.5m (no exceptions)
- GPU OPTIMIZATION: torch.compile, batched inference, max GPU utilization
- VERTICAL AVOIDANCE: Go up/down over obstacles when possible

CORE LOGIC:
- Detect person in frame using YOLO World
- Estimate person's 3D world position from depth map
- Plan A* path from person position to goal
- Visualize path projection on camera view
- Toggle visualization with T key
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|framedrop;1"

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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pygame
from djitellopy import Tello
from ultralytics import YOLOWorld

# Try imports
try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
    print("[PATH] SciPy available for B-spline path smoothing")
except ImportError:
    SCIPY_AVAILABLE = False
    print("[PATH] SciPy not available - B-spline smoothing disabled")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    ONEFORMER_AVAILABLE = True
except ImportError:
    ONEFORMER_AVAILABLE = False

# ==========================================
# RTX 4060 GPU OPTIMIZATION
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Enable CUDA memory optimizations
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM
    # Pre-allocate some memory to reduce fragmentation
    torch.cuda.empty_cache()

# Depth checkpoints
DEPTH_CHECKPOINT_METRIC = 'checkpoints/depth_anything_v2_metric_hypersim_vitb.pth'
DEPTH_CHECKPOINT_RELATIVE = 'checkpoints/depth_anything_v2_vitb.pth'

# Camera Intrinsics (960x720)
FX, FY = 921.17, 919.02
CX, CY = 460.0, 351.0
CAMERA_MATRIX = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

# ==========================================
# V18 COLUMN-BASED CONFIGURATION
# ==========================================
NUM_COLUMNS = 12  # Number of depth columns
COLUMN_HEIGHT_FRACTION = 0.6  # Use middle 60% of frame height

# Distance thresholds (meters)
EMERGENCY_DIST = 0.5    # ABSOLUTE STOP - no exceptions
CRITICAL_DIST = 0.9     # Must turn immediately - increased from 0.8
DANGER_DIST = 1.5       # Slow down significantly - increased from 1.2
CAUTION_DIST = 2.5      # Be careful - increased from 2.0
SAFE_DIST = 3.5         # Can cruise - increased from 3.0
CLEAR_DIST = 4.5        # Full speed ahead - increased from 4.0

# Speed values
SPEED_STOP = 0
SPEED_CREEP = 8       # Very slow when critical - reduced from 15
SPEED_SLOW = 15       # Slow when danger - reduced from 25
SPEED_CRUISE = 35     # Normal cruising - reduced from 40
SPEED_FAST = 55       # Fast when clear - reduced from 60

# Rotation speeds
YAW_DECISIVE = 60       # Strong rotation for obstacle avoidance
YAW_GENTLE = 30         # Gentle steering
YAW_EXPLORE = 45        # Exploration rotation

# Vertical speeds
VERT_SPEED = 35
VERT_SLOW = 20

# Altitude limits
MAX_ALTITUDE = 1.8
MIN_ALTITUDE = 0.3

# Mapping
VOXEL_RES = 0.10
GRID_SIZE = 200
HEIGHT_LEVELS = 30
CENTER = GRID_SIZE // 2
FLOOR_VOXEL_RES = 0.05
FLOOR_GRID_SIZE = 400
FLOOR_CENTER = FLOOR_GRID_SIZE // 2

# Exploration
VISITED_DECAY = 0.995   # Slowly forget visited areas
EXPLORATION_WEIGHT = 0.3  # How much to weight exploration vs depth

# Stuck detection
STUCK_THRESHOLD = 0.1
STUCK_TIME = 2.5

# Path planning
PATH_UPDATE_INTERVAL = 0.3  # Faster updates for person tracking
DEFAULT_TARGET_DISTANCE = 5.0
PATH_STRING_PULL_ENABLED = True
PATH_BSPLINE_ENABLED = True
PATH_BSPLINE_SAMPLES = 50
PATH_BSPLINE_SMOOTHNESS = 0.0

# Path visualization colors
PATH_LINE_COLOR = (0, 255, 255)  # Yellow
PATH_POINT_COLOR = (0, 255, 0)   # Green
PATH_START_COLOR = (255, 0, 0)   # Blue (person position)
PATH_GOAL_COLOR = (0, 0, 255)    # Red
PATH_LINE_THICKNESS = 3
PATH_POINT_RADIUS = 8

# Person detection
PERSON_CLASS = "person"
PERSON_CONFIDENCE_THRESHOLD = 0.4
PERSON_DEPTH_PERCENTILE = 30  # Use lower percentile for more accurate depth

# Floor classes
FLOOR_CLASS_IDS = {3: "floor", 28: "rug", 6: "road", 11: "sidewalk", 13: "ground"}

# Logging
logging.getLogger('djitellopy').setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


# ==========================================
# PATHFINDING STATUS LOGGER
# ==========================================
@dataclass
class PathfindingStatus:
    """Status information for pathfinding algorithm."""
    person_detected: bool = False
    person_world_pos: Optional[Tuple[float, float, float]] = None
    person_screen_pos: Optional[Tuple[int, int]] = None
    person_depth: float = 0.0
    goal_set: bool = False
    goal_world_pos: Optional[Tuple[float, float]] = None
    path_found: bool = False
    path_length: int = 0
    path_distance: float = 0.0
    astar_iterations: int = 0
    string_pull_applied: bool = False
    bspline_applied: bool = False
    last_update_time: float = 0.0
    computation_time_ms: float = 0.0
    error_message: str = ""
    
    def get_status_lines(self) -> List[str]:
        """Get status as list of strings for display."""
        lines = []
        
        # Person status
        if self.person_detected:
            lines.append(f"👤 Person: DETECTED")
            if self.person_world_pos:
                px, py, pz = self.person_world_pos
                lines.append(f"   World: ({px:.2f}, {py:.2f}, {pz:.2f})")
            lines.append(f"   Depth: {self.person_depth:.2f}m")
        else:
            lines.append("👤 Person: NOT FOUND")
        
        # Goal status
        if self.goal_set and self.goal_world_pos:
            gx, gz = self.goal_world_pos
            lines.append(f"🎯 Goal: ({gx:.2f}, {gz:.2f})")
        else:
            lines.append("🎯 Goal: NOT SET")
        
        # Path status
        if self.path_found:
            lines.append(f"📍 Path: {self.path_length} waypoints")
            lines.append(f"   Distance: {self.path_distance:.2f}m")
            lines.append(f"   A* iters: {self.astar_iterations}")
            opts = []
            if self.string_pull_applied:
                opts.append("StringPull")
            if self.bspline_applied:
                opts.append("B-Spline")
            if opts:
                lines.append(f"   Opts: {', '.join(opts)}")
        else:
            lines.append("📍 Path: NO PATH")
        
        # Timing
        lines.append(f"⏱️ Compute: {self.computation_time_ms:.1f}ms")
        
        # Error
        if self.error_message:
            lines.append(f"⚠️ {self.error_message}")
        
        return lines


# ==========================================
# COLUMN DEPTH DATA STRUCTURE
# ==========================================
@dataclass
class ColumnDepthInfo:
    """Depth information for each of the 12 columns."""
    depths: List[float] = field(default_factory=lambda: [10.0] * NUM_COLUMNS)
    min_depth: float = 10.0
    min_column: int = 6  # Center
    max_depth: float = 0.0
    max_column: int = 6  # Center
    center_depth: float = 10.0
    left_avg: float = 10.0
    right_avg: float = 10.0
    best_direction: str = "forward"  # "left", "right", "forward", "blocked"
    recommended_yaw: int = 0
    recommended_speed: int = SPEED_CRUISE
    is_emergency: bool = False
    is_blocked: bool = False


# ==========================================
# A* PATHFINDING WITH STRING PULLING + B-SPLINE
# ==========================================
class AStarPathfinder:
    """
    A* pathfinding with:
    1. A* search for initial path
    2. String Pulling: Remove unnecessary waypoints
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
        self.direction_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        
        # Statistics for logging
        self.last_iterations = 0
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def find_path(self, floor_grid: np.ndarray, start: Tuple[int, int], 
                  goal: Tuple[int, int], obstacle_grid: np.ndarray = None) -> Tuple[List[Tuple[int, int]], bool, bool]:
        """
        Find path from start to goal with string pulling and B-spline smoothing.
        Returns: (path, string_pull_applied, bspline_applied)
        """
        string_pull_applied = False
        bspline_applied = False
        
        if not self._is_valid(start, floor_grid, obstacle_grid):
            start = self._find_nearest_valid(start, floor_grid, obstacle_grid)
            if start is None:
                return [], False, False
        
        if not self._is_valid(goal, floor_grid, obstacle_grid):
            goal = self._find_nearest_valid(goal, floor_grid, obstacle_grid)
            if goal is None:
                return [], False, False
        
        # Step 1: A* search
        raw_path = self._astar_search(floor_grid, start, goal, obstacle_grid)
        if not raw_path:
            return [], False, False
        
        # Step 2: String Pulling
        if PATH_STRING_PULL_ENABLED and len(raw_path) > 2:
            pulled_path = self._string_pull(raw_path, floor_grid, obstacle_grid)
            string_pull_applied = len(pulled_path) < len(raw_path)
        else:
            pulled_path = raw_path
        
        # Step 3: B-Spline smoothing
        if PATH_BSPLINE_ENABLED and SCIPY_AVAILABLE and len(pulled_path) > 3:
            smoothed_path = self._bspline_smooth(pulled_path)
            bspline_applied = True
        else:
            smoothed_path = pulled_path
        
        return smoothed_path, string_pull_applied, bspline_applied
    
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
                self.last_iterations = iterations
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
        
        self.last_iterations = iterations
        return []
    
    def _string_pull(self, path: List[Tuple[int, int]], floor_grid: np.ndarray,
                     obstacle_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Remove unnecessary waypoints using line-of-sight checks."""
        if len(path) <= 2:
            return path
        
        pulled = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
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
        """B-Spline smoothing for rounded corners."""
        if len(path) < 4:
            return path
        
        try:
            x = np.array([p[0] for p in path], dtype=float)
            z = np.array([p[1] for p in path], dtype=float)
            
            tck, u = interpolate.splprep([x, z], s=PATH_BSPLINE_SMOOTHNESS, k=min(3, len(path)-1))
            u_new = np.linspace(0, 1, num_samples)
            smooth_coords = interpolate.splev(u_new, tck)
            
            smoothed_path = []
            for sx, sz in zip(smooth_coords[0], smooth_coords[1]):
                gx = int(np.clip(round(sx), 0, self.grid_size - 1))
                gz = int(np.clip(round(sz), 0, self.grid_size - 1))
                if not smoothed_path or (gx, gz) != smoothed_path[-1]:
                    smoothed_path.append((gx, gz))
            
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
        """Bresenham's line algorithm for line-of-sight check."""
        x1, z1 = p1
        x2, z2 = p2
        dx, dz = abs(x2 - x1), abs(z2 - z1)
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
# PERSON POSITION ESTIMATOR
# ==========================================
class PersonPositionEstimator:
    """
    Estimates 3D world position of detected person from:
    1. 2D bounding box center
    2. Depth map
    3. Camera intrinsics
    """
    
    def __init__(self):
        self.fx = FX
        self.fy = FY
        self.cx = CX
        self.cy = CY
        
        # Smoothing for stable position
        self.position_history = deque(maxlen=5)
        self.last_valid_position = None
    
    def estimate_world_position(self, bbox: Tuple[float, float, float, float],
                                 depth_meters: np.ndarray,
                                 drone_x: float, drone_y: float, drone_z: float,
                                 drone_yaw: float) -> Optional[Tuple[float, float, float]]:
        """
        Estimate person's world position from bounding box and depth.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            depth_meters: Depth map in meters
            drone_x, drone_y, drone_z: Drone position
            drone_yaw: Drone yaw angle in degrees
            
        Returns:
            (world_x, world_y, world_z) or None if estimation fails
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_meters.shape
        
        # Get bounding box center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # Clamp to image bounds
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        
        # Get depth at center region (use percentile for robustness)
        # Sample a region around the center
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        
        roi_x1 = max(0, cx - margin_x)
        roi_x2 = min(w, cx + margin_x)
        roi_y1 = max(0, cy - margin_y)
        roi_y2 = min(h, cy + margin_y)
        
        depth_roi = depth_meters[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if depth_roi.size == 0:
            return None
        
        # Use percentile to filter outliers
        person_depth = float(np.percentile(depth_roi, PERSON_DEPTH_PERCENTILE))
        
        if person_depth < 0.3 or person_depth > 15.0:
            return None
        
        # Convert 2D + depth to 3D camera coordinates
        # Using pinhole camera model
        cam_x = (cx - self.cx) * person_depth / self.fx
        cam_y = (cy - self.cy) * person_depth / self.fy
        cam_z = person_depth
        
        # Transform to world coordinates
        yaw_rad = np.radians(-drone_yaw)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Rotate camera coordinates to world frame
        world_x = cam_x * c - cam_z * s + drone_x
        world_z = cam_x * s + cam_z * c + drone_z
        world_y = drone_y - cam_y  # Y is up, camera Y is down
        
        # Apply temporal smoothing
        self.position_history.append((world_x, world_y, world_z))
        
        if len(self.position_history) >= 2:
            positions = np.array(list(self.position_history))
            smoothed = np.mean(positions, axis=0)
            result = (float(smoothed[0]), float(smoothed[1]), float(smoothed[2]))
        else:
            result = (world_x, world_y, world_z)
        
        self.last_valid_position = result
        return result
    
    def get_floor_position(self, world_pos: Tuple[float, float, float]) -> Tuple[float, float]:
        """Project world position to floor plane (y=0)."""
        return (world_pos[0], world_pos[2])


# ==========================================
# PATH PROJECTION ENGINE
# ==========================================
class PathProjectionEngine:
    """Projects 3D world path onto 2D camera view."""
    
    def __init__(self):
        self.camera_matrix = CAMERA_MATRIX
    
    def world_to_camera(self, world_points: np.ndarray, 
                        drone_x: float, drone_y: float, drone_z: float, 
                        drone_yaw: float) -> np.ndarray:
        """Transform world coordinates to camera coordinates."""
        points = world_points.copy()
        points[:, 0] -= drone_x
        points[:, 1] -= drone_y
        points[:, 2] -= drone_z
        
        yaw_rad = np.radians(drone_yaw)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
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
        """Project camera coordinates to image coordinates."""
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
                     drone_yaw: float, floor_height: float = 0.0,
                     image_width: int = 960, image_height: int = 720) -> Tuple[List[Tuple[int, int]], List[bool]]:
        """Project a 2D floor path to image coordinates."""
        if not world_path:
            return [], []
        
        world_points_3d = np.array([[wx, floor_height, wz] for wx, wz in world_path])
        camera_points = self.world_to_camera(world_points_3d, drone_x, drone_y, drone_z, drone_yaw)
        image_points, valid_mask = self.project_to_image(camera_points, image_width, image_height)
        image_points_int = [(int(p[0]), int(p[1])) for p in image_points]
        
        return image_points_int, valid_mask.tolist()
    
    def draw_path_on_image(self, image: np.ndarray,
                           image_points: List[Tuple[int, int]],
                           valid_mask: List[bool],
                           person_screen_pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Draw projected path on image with person marker."""
        if not image_points:
            return image
        
        result = image.copy()
        valid_points = [(i, p) for i, (p, v) in enumerate(zip(image_points, valid_mask)) if v]
        
        if not valid_points:
            return result
        
        # Draw path lines
        for i in range(len(valid_points) - 1):
            idx1, pt1 = valid_points[i]
            idx2, pt2 = valid_points[i + 1]
            thickness = max(2, min(8, int(8 - i * 0.5)))
            cv2.line(result, pt1, pt2, PATH_LINE_COLOR, thickness)
        
        # Draw waypoints
        for i, (idx, pt) in enumerate(valid_points):
            radius = max(3, min(10, int(10 - i * 0.3)))
            cv2.circle(result, pt, radius, PATH_POINT_COLOR, -1)
            cv2.circle(result, pt, radius + 2, (255, 255, 255), 1)
        
        # Draw person start marker (first point)
        if valid_points:
            _, start_pt = valid_points[0]
            cv2.circle(result, start_pt, PATH_POINT_RADIUS + 4, PATH_START_COLOR, -1)
            cv2.putText(result, "PERSON", (start_pt[0] + 15, start_pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw goal marker (last point)
        if len(valid_points) > 1:
            _, goal_pt = valid_points[-1]
            cv2.circle(result, goal_pt, PATH_POINT_RADIUS + 4, PATH_GOAL_COLOR, -1)
            cv2.putText(result, "GOAL", (goal_pt[0] + 15, goal_pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw person bounding box marker if provided
        if person_screen_pos:
            px, py = person_screen_pos
            cv2.drawMarker(result, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
        
        return result


# ==========================================
# GPU-OPTIMIZED DEPTH ENGINE
# ==========================================
class GPUDepthEngine:
    """
    Optimized depth estimation for RTX 4060.
    Uses torch.compile, batched processing, and keeps data on GPU.
    """
    
    def __init__(self):
        print("[DEPTH] Initializing GPU-optimized depth engine...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.is_metric = self._load_model()
        
        # Normalization tensors on GPU
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # Pre-allocate GPU tensors for inference
        self.input_size = 518
        self.input_buffer = torch.zeros(1, 3, self.input_size, self.input_size, 
                                        device=self.device, dtype=torch.float16)
        
        # Temporal smoothing on GPU
        self.depth_history = deque(maxlen=3)
        
        # Warm up the model
        self._warmup()
        
        print(f"[DEPTH] Engine ready on {self.device}")
        if self.device == 'cuda':
            print(f"[DEPTH] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[DEPTH] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def _load_model(self):
        model_configs = {
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
        }
        config = model_configs['vitb'].copy()
        
        use_metric = os.path.exists(DEPTH_CHECKPOINT_METRIC)
        
        if use_metric:
            checkpoint_path = DEPTH_CHECKPOINT_METRIC
            try:
                metric_depth_path = os.path.join(os.getcwd(), 'metric_depth')
                if os.path.exists(metric_depth_path) and metric_depth_path not in sys.path:
                    sys.path.insert(0, metric_depth_path)
                from depth_anything_v2.dpt import DepthAnythingV2
                import inspect
                sig = inspect.signature(DepthAnythingV2.__init__)
                if 'max_depth' in sig.parameters:
                    config['max_depth'] = 20
                else:
                    use_metric = False
            except ImportError:
                from depth_anything_v2.dpt import DepthAnythingV2
                use_metric = False
        else:
            checkpoint_path = DEPTH_CHECKPOINT_RELATIVE
            from depth_anything_v2.dpt import DepthAnythingV2
        
        model = DepthAnythingV2(**config)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(self.device).eval()
        
        if self.device == 'cuda':
            model = model.half()
            # torch.compile requires Triton which is not available on Windows
            # Skip compilation on Windows or if Triton is missing
            import platform
            if platform.system() != 'Windows':
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    print("[DEPTH] Model compiled with torch.compile")
                except Exception as e:
                    print(f"[DEPTH] torch.compile not available (Triton missing), using eager mode: {e}")
            else:
                print("[DEPTH] Skipping torch.compile on Windows (Triton not supported)")
        
        return model, use_metric
    
    def _warmup(self):
        """Warm up the model with dummy inference."""
        print("[DEPTH] Warming up GPU...")
        dummy = torch.randn(1, 3, self.input_size, self.input_size, 
                           device=self.device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy)
        torch.cuda.synchronize()
        print("[DEPTH] GPU warm-up complete")
    
    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run depth inference optimized for GPU."""
        h, w = frame.shape[:2]
        
        # Resize on CPU (faster for small images)
        img = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert to tensor and move to GPU in one operation
        img_t = torch.from_numpy(img).to(self.device, dtype=torch.float16, non_blocking=True)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Normalize on GPU
        img_t = (img_t - self.mean.half()) / self.std.half()
        
        # Inference
        depth = self.model(img_t)
        
        # Resize on GPU
        depth = F.interpolate(depth[:, None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        
        # Convert to meters
        if self.is_metric:
            depth_meters = depth.float()
        else:
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = (depth - d_min) / (d_max - d_min)
                depth_norm = 1.0 - depth_norm
                depth_meters = 0.5 + depth_norm * 5.5
            else:
                depth_meters = torch.full_like(depth, 3.0, dtype=torch.float32)
        
        depth_meters = torch.clamp(depth_meters, 0.1, 20.0)
        
        # Move to CPU for numpy operations
        depth_np = depth_meters.cpu().numpy()
        
        # Temporal smoothing
        self.depth_history.append(depth_np.copy())
        if len(self.depth_history) >= 2:
            depth_np = np.mean(list(self.depth_history), axis=0)
        
        # Visualization
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 0.1:
            depth_vis = ((1.0 - (depth_np - d_min) / (d_max - d_min)) * 255).astype(np.uint8)
        else:
            depth_vis = np.full_like(depth_np, 128, dtype=np.uint8)
        
        return depth_vis, depth_np.astype(np.float32)


# ==========================================
# V18 COLUMN DEPTH ANALYZER
# ==========================================
class ColumnDepthAnalyzer:
    """
    Analyzes depth in 12 columns and makes decisive navigation decisions.
    This is the CORE of the obstacle avoidance system.
    """
    
    def __init__(self):
        self.history = deque(maxlen=5)
        self.emergency_count = 0
    
    def analyze(self, depth_meters: np.ndarray) -> ColumnDepthInfo:
        """
        Analyze depth map in 12 columns.
        Returns navigation recommendation based on clearest path.
        """
        h, w = depth_meters.shape
        info = ColumnDepthInfo()
        
        # Define analysis region (middle 60% vertically)
        y_start = int(h * 0.2)
        y_end = int(h * 0.8)
        region = depth_meters[y_start:y_end, :]
        
        # Calculate column width
        col_width = w // NUM_COLUMNS
        
        # Analyze each column
        for i in range(NUM_COLUMNS):
            x_start = i * col_width
            x_end = (i + 1) * col_width if i < NUM_COLUMNS - 1 else w
            
            col_data = region[:, x_start:x_end]
            
            # Use 10th percentile for safety (catches thin obstacles)
            col_depth = float(np.percentile(col_data, 10))
            info.depths[i] = col_depth
            
            # Track min/max
            if col_depth < info.min_depth:
                info.min_depth = col_depth
                info.min_column = i
            if col_depth > info.max_depth:
                info.max_depth = col_depth
                info.max_column = i
        
        # Calculate center depth (middle 4 columns)
        center_start = NUM_COLUMNS // 2 - 2
        center_end = NUM_COLUMNS // 2 + 2
        info.center_depth = np.mean(info.depths[center_start:center_end])
        
        # Calculate left/right averages
        info.left_avg = np.mean(info.depths[:NUM_COLUMNS // 3])
        info.right_avg = np.mean(info.depths[2 * NUM_COLUMNS // 3:])
        
        # ===== DECISIVE NAVIGATION LOGIC =====
        
        # EMERGENCY: Any column < 0.5m = STOP
        if info.min_depth < EMERGENCY_DIST:
            info.is_emergency = True
            info.is_blocked = True
            info.recommended_speed = SPEED_STOP
            # Rotate away from closest obstacle
            if info.min_column < NUM_COLUMNS // 2:
                info.recommended_yaw = YAW_DECISIVE  # Turn right
                info.best_direction = "right"
            else:
                info.recommended_yaw = -YAW_DECISIVE  # Turn left
                info.best_direction = "left"
            return info
        
        # CRITICAL: Center blocked < 0.8m = Strong rotation
        if info.center_depth < CRITICAL_DIST:
            info.is_blocked = True
            info.recommended_speed = SPEED_CREEP
            
            # Find best escape direction based on max depth column
            if info.max_column < NUM_COLUMNS // 2:
                # Best path is on the left
                info.recommended_yaw = -YAW_DECISIVE
                info.best_direction = "left"
            else:
                # Best path is on the right
                info.recommended_yaw = YAW_DECISIVE
                info.best_direction = "right"
            return info
        
        # DANGER: Center < 1.2m = Moderate rotation + slow speed
        if info.center_depth < DANGER_DIST:
            info.recommended_speed = SPEED_SLOW
            
            # Steer towards clearer side
            if info.left_avg > info.right_avg + 0.3:
                info.recommended_yaw = -YAW_GENTLE
                info.best_direction = "left"
            elif info.right_avg > info.left_avg + 0.3:
                info.recommended_yaw = YAW_GENTLE
                info.best_direction = "right"
            else:
                # Go towards max depth column
                col_offset = info.max_column - (NUM_COLUMNS // 2)
                info.recommended_yaw = int(col_offset * 8)  # Scale to yaw
                info.best_direction = "forward"
            return info
        
        # CAUTION: Center < 2.0m = Gentle steering but slower speed
        if info.center_depth < CAUTION_DIST:
            info.recommended_speed = SPEED_SLOW  # Use SLOW speed in caution zone
            
            # Gentle correction towards clearer path
            col_offset = info.max_column - (NUM_COLUMNS // 2)
            info.recommended_yaw = int(col_offset * 5)
            info.best_direction = "forward"
            return info
        
        # SAFE: Center >= 2.0m
        if info.center_depth < SAFE_DIST:
            info.recommended_speed = SPEED_CRUISE
        else:
            info.recommended_speed = SPEED_FAST
        
        # Minor steering towards furthest depth
        col_offset = info.max_column - (NUM_COLUMNS // 2)
        info.recommended_yaw = int(col_offset * 3)
        info.best_direction = "forward"
        
        return info
    
    def get_column_visualization(self, frame: np.ndarray, depth_meters: np.ndarray, 
                                  info: ColumnDepthInfo) -> np.ndarray:
        """
        Draw column depth visualization on frame.
        Shows 12 columns with depth values and colors.
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        col_width = w // NUM_COLUMNS
        
        # Draw column overlays
        for i in range(NUM_COLUMNS):
            x_start = i * col_width
            x_end = (i + 1) * col_width if i < NUM_COLUMNS - 1 else w
            
            depth = info.depths[i]
            
            # Color based on depth
            if depth < EMERGENCY_DIST:
                color = (0, 0, 255)  # Red - DANGER
                alpha = 0.5
            elif depth < CRITICAL_DIST:
                color = (0, 128, 255)  # Orange
                alpha = 0.4
            elif depth < DANGER_DIST:
                color = (0, 255, 255)  # Yellow
                alpha = 0.3
            elif depth < CAUTION_DIST:
                color = (0, 255, 128)  # Light green
                alpha = 0.2
            else:
                color = (0, 255, 0)  # Green - SAFE
                alpha = 0.15
            
            # Draw semi-transparent overlay
            overlay = result.copy()
            cv2.rectangle(overlay, (x_start, 0), (x_end, h), color, -1)
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            # Draw column border
            cv2.line(result, (x_start, 0), (x_start, h), (100, 100, 100), 1)
            
            # Draw depth value at top
            depth_text = f"{depth:.1f}m"
            text_x = x_start + 5
            text_y = 25
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (text_x - 2, text_y - th - 2), 
                         (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)
            cv2.putText(result, depth_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Mark max depth column with arrow
            if i == info.max_column:
                arrow_x = (x_start + x_end) // 2
                cv2.arrowedLine(result, (arrow_x, h - 50), (arrow_x, h - 20), 
                               (0, 255, 255), 3, tipLength=0.5)
                cv2.putText(result, "BEST", (arrow_x - 20, h - 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw navigation recommendation
        nav_text = f"NAV: {info.best_direction.upper()} | YAW: {info.recommended_yaw} | SPD: {info.recommended_speed}"
        cv2.putText(result, nav_text, (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Emergency warning
        if info.is_emergency:
            cv2.putText(result, "!!! EMERGENCY STOP !!!", (w // 2 - 150, h // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif info.is_blocked:
            cv2.putText(result, "BLOCKED - ROTATING", (w // 2 - 120, h // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
        
        return result


# ==========================================
# SMART EXPLORATION ENGINE
# ==========================================
class SmartExplorer:
    """
    Handles exploration logic:
    1. Avoids re-visiting areas
    2. Prioritizes furthest depth
    3. Maintains exploration grid
    """
    
    def __init__(self):
        self.visited_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.last_pos = (0.0, 0.0, 0.0)
        self.last_move_time = time.time()
        self.explore_direction = 1  # 1 = right preference, -1 = left preference
        self.stuck_count = 0
    
    def update_visited(self, pos_x: float, pos_z: float):
        """Mark current position as visited."""
        ix = int(pos_x / VOXEL_RES + CENTER)
        iz = int(pos_z / VOXEL_RES + CENTER)
        
        # Mark a radius around current position
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                nx, nz = ix + dx, iz + dz
                if 0 <= nx < GRID_SIZE and 0 <= nz < GRID_SIZE:
                    dist = math.sqrt(dx*dx + dz*dz)
                    self.visited_grid[nz, nx] = min(1.0, self.visited_grid[nz, nx] + 0.3 / (dist + 1))
        
        # Decay old visits
        self.visited_grid *= VISITED_DECAY
    
    def is_stuck(self, pos_x: float, pos_y: float, pos_z: float) -> bool:
        """Check if drone is stuck (not moving)."""
        dist = math.sqrt((pos_x - self.last_pos[0])**2 + 
                        (pos_y - self.last_pos[1])**2 + 
                        (pos_z - self.last_pos[2])**2)
        
        if dist < STUCK_THRESHOLD:
            if time.time() - self.last_move_time > STUCK_TIME:
                self.stuck_count += 1
                return True
        else:
            self.last_pos = (pos_x, pos_y, pos_z)
            self.last_move_time = time.time()
            self.stuck_count = 0
        
        return False
    
    def get_exploration_direction(self, column_info: ColumnDepthInfo, yaw: float) -> int:
        """
        Get exploration yaw adjustment.
        Prioritizes: 1) Furthest depth, 2) Unvisited areas
        """
        # Start with column-based recommendation
        base_yaw = column_info.recommended_yaw
        
        # If we're going to a visited area, try to adjust
        # This is a simple heuristic - rotate away from visited areas
        
        # Calculate visited density in direction we're heading
        # For now, use the max depth column as the exploration target
        return base_yaw
    
    def get_unstuck_action(self) -> Tuple[int, int, int, int]:
        """Return action to get unstuck."""
        self.explore_direction *= -1  # Alternate direction
        self.last_move_time = time.time()
        
        # Rotate in place
        return (0, 0, 0, YAW_EXPLORE * self.explore_direction)


# ==========================================
# FLOOR SEGMENTATION (GPU-OPTIMIZED)
# ==========================================
class FloorSegmentationEngine:
    """OneFormer floor segmentation with GPU optimization."""
    
    def __init__(self):
        if not ONEFORMER_AVAILABLE:
            self.enabled = False
            return
        
        print("[SEG] Initializing OneFormer (GPU-optimized)...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_tiny", local_files_only=True)
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_tiny", local_files_only=True).to(self.device)
            
            if self.device == 'cuda':
                self.model = self.model.half()
            
            self.model.eval()
            self.enabled = True
            print(f"[SEG] OneFormer ready on {self.device}")
        except Exception as e:
            print(f"[SEG] Failed: {e}")
            self.enabled = False
    
    @torch.inference_mode()
    def segment_floor(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return np.zeros(frame.shape[:2], dtype=np.uint8), frame.copy()
        
        h, w = frame.shape[:2]
        scale = 512 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=frame_rgb, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        
        if self.device == 'cuda':
            for k, v in inputs.items():
                if v.dtype == torch.float32:
                    inputs[k] = v.half()
        
        outputs = self.model(**inputs)
        predicted_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(new_h, new_w)])[0]
        
        seg_map = predicted_map.cpu().numpy()
        
        floor_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        for class_id in FLOOR_CLASS_IDS.keys():
            floor_mask[seg_map == class_id] = 255
        
        floor_mask = cv2.resize(floor_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create visualization
        colors = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for class_id in FLOOR_CLASS_IDS.keys():
            colors[seg_map == class_id] = [0, 255, 0]
        colors[seg_map == 0] = [255, 100, 100]
        colors[seg_map == 5] = [255, 0, 255]
        
        blended = cv2.addWeighted(frame_rgb, 0.5, colors, 0.5, 0)
        blended = cv2.resize(blended, (w, h))
        seg_vis = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        
        return floor_mask, seg_vis


# ==========================================
# VISUAL ODOMETRY
# ==========================================
class VisualOdometry:
    """Simple visual odometry with IMU fusion."""
    
    def __init__(self):
        self.K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        self.prev_gray = None
        self.prev_points = None
        
        self.pos_x = 0.0
        self.pos_y = 1.0
        self.pos_z = 0.0
        self.yaw = 0.0
        
    def update(self, frame: np.ndarray, imu_yaw: float, imu_height: Optional[float] = None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.yaw = float(imu_yaw)
        
        if imu_height is not None and imu_height > 0:
            self.pos_y = 0.9 * self.pos_y + 0.1 * imu_height
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200, 
                                                       qualityLevel=0.15, minDistance=5)
            return
        
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200,
                                                       qualityLevel=0.15, minDistance=5)
            return
        
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None,
            winSize=(21, 21), maxLevel=3)
        
        if p1 is not None:
            good_new = p1[status.flatten() == 1]
            good_old = self.prev_points[status.flatten() == 1]
            
            if len(good_new) > 20:
                flow = good_new - good_old
                mag = np.median(np.linalg.norm(flow, axis=1))
                
                if mag > 2.0:
                    try:
                        E, mask = cv2.findEssentialMat(good_new, good_old, self.K,
                                                       method=cv2.RANSAC, prob=0.999, threshold=0.5)
                        if E is not None:
                            _, R, t, _ = cv2.recoverPose(E, good_new, good_old, self.K, mask=mask)
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
                    except:
                        pass
                
                self.prev_points = good_new.reshape(-1, 1, 2)
            else:
                self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200,
                                                           qualityLevel=0.15, minDistance=5)
        
        self.prev_gray = gray


# ==========================================
# FLOOR MAP & PATH PLANNING (PERSON-TO-GOAL)
# ==========================================
class FloorMap3D:
    """3D floor map with A* path planning from PERSON to GOAL."""
    
    def __init__(self):
        self.floor_grid = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE), dtype=np.float32)
        self.floor_confidence = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE), dtype=np.float32)
        self.obstacle_grid = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE), dtype=np.float32)
        self.floor_points = []
        self.floor_colors = []
        self.scene_points = []
        self.scene_colors = []
        
        # Path planning - PERSON to GOAL
        self.pathfinder = AStarPathfinder(FLOOR_GRID_SIZE)
        self.goal_position: Optional[Tuple[int, int]] = None  # Goal grid position
        self.goal_world_pos: Optional[Tuple[float, float]] = None  # Goal world position
        self.person_position: Optional[Tuple[int, int]] = None  # Person grid position
        self.person_world_pos: Optional[Tuple[float, float]] = None  # Person world position
        self.current_path: List[Tuple[int, int]] = []
        self.world_path: List[Tuple[float, float]] = []
        
        self.lock = threading.Lock()
        print("[FLOOR MAP] Initialized with Person-to-Goal A* pathfinding")
    
    def update(self, depth_meters: np.ndarray, floor_mask: np.ndarray,
               pos_x: float, pos_y: float, pos_z: float, yaw: float):
        h, w = depth_meters.shape
        
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
            
            ix = (floor_x / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            iz = (floor_z / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            
            valid_grid = ((ix >= 0) & (ix < FLOOR_GRID_SIZE) & 
                         (iz >= 0) & (iz < FLOOR_GRID_SIZE))
            
            for i, j in zip(ix[valid_grid], iz[valid_grid]):
                self.floor_grid[j, i] = 1.0
                self.floor_confidence[j, i] = min(self.floor_confidence[j, i] + 0.1, 1.0)
            
            obs_x = x_w[~is_floor]
            obs_z = z_w[~is_floor]
            obs_y = y_w[~is_floor]
            obs_height_valid = (obs_y > 0.1) & (obs_y < 2.0)
            obs_x = obs_x[obs_height_valid]
            obs_z = obs_z[obs_height_valid]
            
            obs_ix = (obs_x / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            obs_iz = (obs_z / FLOOR_VOXEL_RES + FLOOR_CENTER).astype(np.int32)
            
            valid_obs = ((obs_ix >= 0) & (obs_ix < FLOOR_GRID_SIZE) & 
                        (obs_iz >= 0) & (obs_iz < FLOOR_GRID_SIZE))
            
            for i, j in zip(obs_ix[valid_obs], obs_iz[valid_obs]):
                self.obstacle_grid[j, i] = min(self.obstacle_grid[j, i] + 0.1, 1.0)
    
    def set_goal(self, world_x: float, world_z: float):
        """Set goal position in world coordinates."""
        ix = int(world_x / FLOOR_VOXEL_RES + FLOOR_CENTER)
        iz = int(world_z / FLOOR_VOXEL_RES + FLOOR_CENTER)
        self.goal_position = (ix, iz)
        self.goal_world_pos = (world_x, world_z)
        print(f"[PATH] Goal set at world ({world_x:.2f}, {world_z:.2f}) -> grid ({ix}, {iz})")
    
    def set_person_position(self, world_x: float, world_z: float):
        """Set person position in world coordinates."""
        ix = int(world_x / FLOOR_VOXEL_RES + FLOOR_CENTER)
        iz = int(world_z / FLOOR_VOXEL_RES + FLOOR_CENTER)
        self.person_position = (ix, iz)
        self.person_world_pos = (world_x, world_z)
    
    def find_path_from_person(self) -> Tuple[List[Tuple[float, float]], bool, bool, int]:
        """
        Find path from person position to goal.
        Returns: (world_path, string_pull_applied, bspline_applied, iterations)
        """
        if self.person_position is None or self.goal_position is None:
            return [], False, False, 0
        
        with self.lock:
            grid_path, string_pull, bspline = self.pathfinder.find_path(
                self.floor_grid,
                self.person_position,
                self.goal_position,
                (self.obstacle_grid > 0.5).astype(np.float32)
            )
        
        if not grid_path:
            self.current_path = []
            self.world_path = []
            return [], False, False, self.pathfinder.last_iterations
        
        # Convert grid path to world path
        world_path = []
        for gx, gz in grid_path:
            wx = (gx - FLOOR_CENTER) * FLOOR_VOXEL_RES
            wz = (gz - FLOOR_CENTER) * FLOOR_VOXEL_RES
            world_path.append((wx, wz))
        
        self.current_path = grid_path
        self.world_path = world_path
        return world_path, string_pull, bspline, self.pathfinder.last_iterations
    
    def get_path_distance(self) -> float:
        """Calculate total path distance in meters."""
        if len(self.world_path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self.world_path) - 1):
            x1, z1 = self.world_path[i]
            x2, z2 = self.world_path[i + 1]
            total += math.sqrt((x2 - x1)**2 + (z2 - z1)**2)
        return total
    
    def get_visualization_2d(self) -> np.ndarray:
        vis = np.zeros((FLOOR_GRID_SIZE, FLOOR_GRID_SIZE, 3), dtype=np.uint8)
        with self.lock:
            vis[self.floor_grid > 0] = [0, 150, 0]
            vis[self.floor_confidence > 0.5] = [0, 255, 0]
            vis[self.obstacle_grid > 0.5] = [0, 0, 200]
            
            # Draw path
            for gx, gz in self.current_path:
                if 0 <= gx < FLOOR_GRID_SIZE and 0 <= gz < FLOOR_GRID_SIZE:
                    cv2.circle(vis, (gx, gz), 2, (0, 255, 255), -1)
            
            # Draw person position (start of path)
            if self.person_position:
                px, pz = self.person_position
                if 0 <= px < FLOOR_GRID_SIZE and 0 <= pz < FLOOR_GRID_SIZE:
                    cv2.circle(vis, (px, pz), 5, (255, 0, 255), -1)  # Magenta for person
            
            # Draw goal
            if self.goal_position:
                tx, tz = self.goal_position
                if 0 <= tx < FLOOR_GRID_SIZE and 0 <= tz < FLOOR_GRID_SIZE:
                    cv2.circle(vis, (tx, tz), 5, (255, 0, 0), -1)  # Blue for goal
        
        return vis


# ==========================================
# OPEN3D VISUALIZER
# ==========================================
class Open3DVisualizer:
    """Real-time 3D visualization."""
    
    def __init__(self, floor_map: FloorMap3D):
        if not OPEN3D_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        self.floor_map = floor_map
        self.running = False
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.lock = threading.Lock()
    
    def start(self):
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def update_drone_position(self, x, y, z, yaw):
        with self.lock:
            self.drone_pos = np.array([x, y, z])
    
    def _run(self):
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window("Floor Map 3D", width=800, height=600)
            
            pcd = o3d.geometry.PointCloud()
            drone_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            drone_marker.paint_uniform_color([1, 0, 0])
            
            vis.add_geometry(pcd)
            vis.add_geometry(drone_marker)
            
            opt = vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.15])
            opt.point_size = 3.0
            
            while self.running:
                with self.lock:
                    drone_marker.translate(-drone_marker.get_center())
                    drone_marker.translate(self.drone_pos)
                
                vis.update_geometry(drone_marker)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)
            
            vis.destroy_window()
        except Exception as e:
            print(f"[3D VIS] Error: {e}")
            self.enabled = False


# ==========================================
# THREADED CAMERA
# ==========================================
class ThreadedCam:
    def __init__(self, src: str = "udp://@0.0.0.0:11111"):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.lock = threading.Lock()
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()

    def start(self):
        if self.started:
            return self
        self.started = True
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.grabbed, self.frame = grabbed, frame
                    self.frame_count += 1
                    if time.time() - self.fps_time > 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.fps_time = time.time()
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_fps(self):
        with self.lock:
            return self.fps

    def stop(self):
        self.started = False
        self.cap.release()


# ==========================================
# V18.1 PERSON-TO-GOAL BRAIN
# ==========================================
class DecisiveBrain:
    """
    V18.1 Brain: Person-to-Goal path planning with decisive navigation.
    
    Core logic:
    1. Always detect PERSON using YOLO World
    2. Estimate person's 3D world position from depth
    3. Plan A* path from PERSON to GOAL
    4. Visualize path projection on camera
    5. Standard obstacle avoidance for drone
    """
    
    def __init__(self, target_obj: str):
        print(f"[BRAIN V18.1] Initializing for target: {target_obj}")
        
        # Core engines
        self.depth_engine = GPUDepthEngine()
        self.column_analyzer = ColumnDepthAnalyzer()
        self.explorer = SmartExplorer()
        self.vo = VisualOdometry()
        self.floor_engine = FloorSegmentationEngine()
        self.floor_map = FloorMap3D()
        self.path_projector = PathProjectionEngine()
        self.person_estimator = PersonPositionEstimator()
        
        # 3D visualization
        self.vis3d = Open3DVisualizer(self.floor_map) if OPEN3D_AVAILABLE else None
        
        # YOLO for person detection (always on) + target detection
        self.yolo = YOLOWorld('yolov8s-world.pt')
        # Set classes: always detect person, plus the target object
        self.target_obj = target_obj
        self.yolo.set_classes([PERSON_CLASS, target_obj])
        self.yolo.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State
        self.mode = "INIT"
        self.current_altitude = 1.0
        
        # Segmentation timing
        self.seg_frame_count = 0
        self.last_floor_mask = None
        self.last_seg_vis = None
        
        # Path planning state
        self.last_path_update = 0
        self.goal_set = False
        
        # Path visualization toggle
        self.show_path_visualization = True
        
        # Pathfinding status for logging
        self.path_status = PathfindingStatus()
        
        # Person detection state
        self.last_person_detection = None
        self.person_lost_time = 0
        
        print("[BRAIN V18.1] Ready - Person-to-Goal Path Planning Active")
        print(f"[BRAIN V18.1] Detecting: {PERSON_CLASS} + {target_obj}")
    
    @property
    def pos_x(self): return self.vo.pos_x
    @property
    def pos_y(self): return self.vo.pos_y
    @property
    def pos_z(self): return self.vo.pos_z
    @property
    def yaw(self): return self.vo.yaw
    
    def start_visualization(self):
        if self.vis3d:
            self.vis3d.start()
    
    def stop_visualization(self):
        if self.vis3d:
            self.vis3d.stop()
    
    def toggle_path_visualization(self):
        """Toggle path overlay on/off."""
        self.show_path_visualization = not self.show_path_visualization
        state = "ON" if self.show_path_visualization else "OFF"
        print(f"[PATH VIS] Path visualization: {state}")
    
    def update_vo(self, frame, imu_yaw, imu_height=None):
        self.vo.update(frame, imu_yaw, imu_height)
        if imu_height:
            self.current_altitude = imu_height
        if self.vis3d:
            self.vis3d.update_drone_position(self.pos_x, self.pos_y, self.pos_z, self.yaw)
    
    def set_goal_from_direction(self):
        """Set goal in current facing direction."""
        target_x = self.pos_x + DEFAULT_TARGET_DISTANCE * np.sin(np.radians(-self.yaw))
        target_z = self.pos_z - DEFAULT_TARGET_DISTANCE * np.cos(np.radians(-self.yaw))
        self.floor_map.set_goal(target_x, target_z)
        self.goal_set = True
        self.path_status.goal_set = True
        self.path_status.goal_world_pos = (target_x, target_z)
        print(f"[PATH] Goal set {DEFAULT_TARGET_DISTANCE}m ahead at ({target_x:.2f}, {target_z:.2f})")
    
    def _detect_objects(self, frame: np.ndarray) -> Tuple[Optional[dict], Optional[dict], any]:
        """
        Detect person and target object in frame.
        Returns: (person_info, target_info, all_boxes)
        """
        results = self.yolo.predict(frame, verbose=False, conf=PERSON_CONFIDENCE_THRESHOLD, half=True)
        
        if not results or len(results[0].boxes) == 0:
            return None, None, None
        
        boxes = results[0].boxes
        h, w = frame.shape[:2]
        
        person_info = None
        target_info = None
        
        # Find largest person and target
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = self.yolo.names[cls_id] if hasattr(self.yolo, 'names') else ""
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            
            info = {
                'bbox': (x1, y1, x2, y2),
                'cx': (x1 + x2) / 2 / w,
                'cy': (y1 + y2) / 2 / h,
                'size': area / (w * h),
                'class': cls_name
            }
            
            # Check if it's a person
            if cls_id == 0 or cls_name.lower() == PERSON_CLASS:
                if person_info is None or info['size'] > person_info['size']:
                    person_info = info
            # Check if it's the target object
            elif cls_name.lower() == self.target_obj.lower():
                if target_info is None or info['size'] > target_info['size']:
                    target_info = info
        
        return person_info, target_info, boxes
    
    def update_person_path(self, frame: np.ndarray, depth_meters: np.ndarray, 
                           person_info: Optional[dict]) -> bool:
        """
        Update path from detected person to goal.
        Returns True if path was updated successfully.
        """
        start_time = time.time()
        
        # Reset status
        self.path_status.error_message = ""
        
        if person_info is None:
            self.path_status.person_detected = False
            self.path_status.person_world_pos = None
            self.path_status.person_screen_pos = None
            self.path_status.path_found = False
            
            # Use last known position if recent
            if self.person_lost_time == 0:
                self.person_lost_time = time.time()
            
            if time.time() - self.person_lost_time > 2.0:
                self.path_status.error_message = "Person lost for >2s"
                self.floor_map.current_path = []
                self.floor_map.world_path = []
            
            return False
        
        self.person_lost_time = 0
        self.path_status.person_detected = True
        
        # Get person screen position
        bbox = person_info['bbox']
        screen_cx = int((bbox[0] + bbox[2]) / 2)
        screen_cy = int((bbox[1] + bbox[3]) / 2)
        self.path_status.person_screen_pos = (screen_cx, screen_cy)
        
        # Estimate person's world position
        person_world = self.person_estimator.estimate_world_position(
            bbox, depth_meters, self.pos_x, self.pos_y, self.pos_z, self.yaw
        )
        
        if person_world is None:
            self.path_status.error_message = "Failed to estimate depth"
            return False
        
        self.path_status.person_world_pos = person_world
        
        # Get depth at person
        h, w = depth_meters.shape
        roi_cx = max(0, min(w-1, screen_cx))
        roi_cy = max(0, min(h-1, screen_cy))
        self.path_status.person_depth = float(depth_meters[roi_cy, roi_cx])
        
        # Set person floor position for path planning
        floor_pos = self.person_estimator.get_floor_position(person_world)
        self.floor_map.set_person_position(floor_pos[0], floor_pos[1])
        
        # Check if goal is set
        if not self.goal_set:
            self.path_status.goal_set = False
            self.path_status.error_message = "Goal not set (press G)"
            return False
        
        # Rate limit path updates
        if time.time() - self.last_path_update < PATH_UPDATE_INTERVAL:
            return True
        
        self.last_path_update = time.time()
        
        # Find path from person to goal
        world_path, string_pull, bspline, iterations = self.floor_map.find_path_from_person()
        
        # Update status
        self.path_status.astar_iterations = iterations
        self.path_status.string_pull_applied = string_pull
        self.path_status.bspline_applied = bspline
        
        if world_path:
            self.path_status.path_found = True
            self.path_status.path_length = len(world_path)
            self.path_status.path_distance = self.floor_map.get_path_distance()
        else:
            self.path_status.path_found = False
            self.path_status.path_length = 0
            self.path_status.path_distance = 0.0
            self.path_status.error_message = "No valid path found"
        
        # Timing
        self.path_status.computation_time_ms = (time.time() - start_time) * 1000
        self.path_status.last_update_time = time.time()
        
        return self.path_status.path_found
    
    def get_path_overlay(self, frame: np.ndarray, person_info: Optional[dict]) -> np.ndarray:
        """Draw path overlay on camera frame."""
        if not self.show_path_visualization:
            return frame.copy()
        
        if not self.floor_map.world_path:
            result = frame.copy()
            # Show status even without path
            if not self.goal_set:
                cv2.putText(result, "Press G to set goal", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif not self.path_status.person_detected:
                cv2.putText(result, "Looking for person...", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(result, "Building path...", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return result
        
        # Get person screen position for marker
        person_screen = self.path_status.person_screen_pos
        
        image_points, valid_mask = self.path_projector.project_path(
            self.floor_map.world_path,
            self.pos_x, self.pos_y, self.pos_z, self.yaw,
            floor_height=0.0,
            image_width=frame.shape[1],
            image_height=frame.shape[0]
        )
        
        result = self.path_projector.draw_path_on_image(frame, image_points, valid_mask, person_screen)
        
        # Draw person bounding box
        if person_info:
            x1, y1, x2, y2 = [int(v) for v in person_info['bbox']]
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(result, "PERSON", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return result
    
    def _can_go_over(self, column_info: ColumnDepthInfo) -> bool:
        """Check if obstacle can be overcome by going up."""
        return (column_info.is_blocked and 
                column_info.center_depth > CRITICAL_DIST and 
                self.current_altitude < MAX_ALTITUDE - 0.3)
    
    def _can_go_under(self, column_info: ColumnDepthInfo) -> bool:
        """Check if obstacle can be overcome by going down."""
        return (column_info.is_blocked and 
                self.current_altitude > MIN_ALTITUDE + 0.3)
    
    def get_action(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, any, Tuple[int, int, int, int], np.ndarray, Optional[dict]]:
        """
        Main decision loop.
        Returns: (depth_vis, column_vis, path_overlay, targets, rc_command, seg_vis, person_info)
        """
        # Run depth inference
        depth_vis, depth_meters = self.depth_engine.infer(frame)
        
        # Analyze columns
        column_info = self.column_analyzer.analyze(depth_meters)
        
        # Create column visualization
        column_vis = self.column_analyzer.get_column_visualization(frame, depth_meters, column_info)
        
        # Update explorer
        self.explorer.update_visited(self.pos_x, self.pos_z)
        
        # Run segmentation periodically
        self.seg_frame_count += 1
        if self.seg_frame_count >= 5 or self.last_floor_mask is None:
            self.seg_frame_count = 0
            self.last_floor_mask, self.last_seg_vis = self.floor_engine.segment_floor(frame)
        
        floor_mask = self.last_floor_mask if self.last_floor_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
        seg_vis = self.last_seg_vis if self.last_seg_vis is not None else frame.copy()
        
        # Update floor map
        self.floor_map.update(depth_meters, floor_mask, self.pos_x, self.pos_y, self.pos_z, self.yaw)
        
        # Detect person and target
        person_info, target_info, boxes = self._detect_objects(frame)
        
        # Update person-to-goal path
        self.update_person_path(frame, depth_meters, person_info)
        
        # Get path overlay
        path_overlay = self.get_path_overlay(frame, person_info)
        
        # ===== DECISIVE NAVIGATION =====
        
        lat, fwd, vert, yaw_cmd = 0, 0, 0, 0
        
        # EMERGENCY: Absolute stop
        if column_info.is_emergency:
            self.mode = "🛑 EMERGENCY"
            fwd = SPEED_STOP
            yaw_cmd = column_info.recommended_yaw
            return depth_vis, column_vis, path_overlay, boxes, (0, 0, 0, yaw_cmd), seg_vis, person_info
        
        # Check if stuck
        if self.explorer.is_stuck(self.pos_x, self.pos_y, self.pos_z):
            self.mode = "🔄 UNSTUCK"
            rc = self.explorer.get_unstuck_action()
            return depth_vis, column_vis, path_overlay, boxes, rc, seg_vis, person_info
        
        # TARGET TRACKING (original target object, not person)
        if target_info:
            self.mode = "🎯 TARGET"
            h_err = target_info['cx'] - 0.5
            v_err = target_info['cy'] - 0.5
            
            yaw_cmd = int(h_err * 80)
            vert = int(-v_err * 40)
            
            if abs(h_err) < 0.1 and abs(v_err) < 0.1:
                if column_info.center_depth > CRITICAL_DIST:
                    fwd = min(column_info.recommended_speed, SPEED_SLOW)
            
            return depth_vis, column_vis, path_overlay, boxes, (0, fwd, vert, yaw_cmd), seg_vis, person_info
        
        # BLOCKED: Try vertical escape
        if column_info.is_blocked:
            if self._can_go_over(column_info):
                self.mode = "⬆️ GO OVER"
                vert = VERT_SPEED
                yaw_cmd = column_info.recommended_yaw // 2
                fwd = 0  # Don't move forward while going over
            elif self._can_go_under(column_info):
                self.mode = "⬇️ GO UNDER"
                vert = -VERT_SPEED
                yaw_cmd = column_info.recommended_yaw // 2
                fwd = 0  # Don't move forward while going under
            else:
                self.mode = "🔄 ROTATING"
                yaw_cmd = column_info.recommended_yaw
                fwd = 0  # STOP forward movement while rotating away from obstacle
            
            return depth_vis, column_vis, path_overlay, boxes, (0, fwd, vert, yaw_cmd), seg_vis, person_info
        
        # EXPLORE: Move towards furthest depth
        self.mode = f"🔭 EXPLORE ({column_info.best_direction})"
        
        fwd = column_info.recommended_speed
        yaw_cmd = column_info.recommended_yaw
        
        # Altitude maintenance
        if self.current_altitude > MAX_ALTITUDE - 0.2:
            vert = -VERT_SLOW
        elif self.current_altitude < MIN_ALTITUDE + 0.2:
            vert = VERT_SLOW
        
        return depth_vis, column_vis, path_overlay, boxes, (0, fwd, vert, yaw_cmd), seg_vis, person_info


# ==========================================
# MANUAL CONTROLLER
# ==========================================
class ManualController:
    def get_rc(self, keys):
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
# V18.1 VISUALIZER WITH PATH STATUS LOGGING
# ==========================================
class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 900))
        pygame.display.set_caption("Tello V18.1 - Person-to-Goal Path Planning")
        self.font = pygame.font.Font(None, 26)
        self.font_sm = pygame.font.Font(None, 20)
        self.font_lg = pygame.font.Font(None, 32)
        self.font_xs = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()
    
    def render(self, frame, depth_vis, column_vis, path_overlay, brain, targets, auto, battery, rc, 
               column_info, seg_vis=None, cam_fps=0, person_info=None):
        self.screen.fill((15, 15, 20))
        
        # ===== LEFT TOP: Column Visualization =====
        col_view = cv2.resize(column_vis, (480, 360))
        if targets:
            for b in targets:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                sx, sy = 480/960, 360/720
                cv2.rectangle(col_view, (int(x1*sx), int(y1*sy)), 
                             (int(x2*sx), int(y2*sy)), (0, 255, 0), 2)
        
        surf = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(col_view, cv2.COLOR_BGR2RGB), (1, 0, 2)))
        self.screen.blit(surf, (10, 10))
        self.screen.blit(self.font.render("12-Column Depth Analysis", True, (0, 255, 255)), (10, 375))
        
        # ===== LEFT BOTTOM: Path Overlay View =====
        if path_overlay is not None:
            path_view = cv2.resize(path_overlay, (480, 360))
            surf_path = pygame.surfarray.make_surface(
                np.transpose(cv2.cvtColor(path_view, cv2.COLOR_BGR2RGB), (1, 0, 2)))
            self.screen.blit(surf_path, (10, 400))
            
            vis_state = "ON" if brain.show_path_visualization else "OFF"
            path_count = len(brain.floor_map.world_path)
            path_text = f"Person→Goal Path ({path_count} pts) [T: {vis_state}]"
            self.screen.blit(self.font.render(path_text, True, (0, 255, 255)), (10, 765))
        
        # ===== MIDDLE TOP: Depth Map =====
        d_color = cv2.applyColorMap(cv2.resize(depth_vis, (280, 210)), cv2.COLORMAP_MAGMA)
        if column_info.is_emergency:
            cv2.rectangle(d_color, (0, 0), (279, 209), (0, 0, 255), 6)
        elif column_info.is_blocked:
            cv2.rectangle(d_color, (0, 0), (279, 209), (0, 128, 255), 4)
        
        surf = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB), (1, 0, 2)))
        self.screen.blit(surf, (500, 10))
        self.screen.blit(self.font_sm.render(f"Depth Map | Min: {column_info.min_depth:.2f}m", 
                                              True, (180, 180, 180)), (500, 225))
        
        # ===== MIDDLE: Segmentation =====
        if seg_vis is not None:
            seg_small = cv2.resize(seg_vis, (280, 210))
            surf_seg = pygame.surfarray.make_surface(
                np.transpose(cv2.cvtColor(seg_small, cv2.COLOR_BGR2RGB), (1, 0, 2)))
            self.screen.blit(surf_seg, (500, 250))
            self.screen.blit(self.font_sm.render("Floor Segmentation", True, (180, 180, 180)), (500, 465))
        
        # ===== MIDDLE: Floor Map with Path =====
        floor_vis = brain.floor_map.get_visualization_2d()
        
        # Draw drone position
        rx = np.clip(int(brain.pos_x / VOXEL_RES) + CENTER, 2, GRID_SIZE - 3)
        rz = np.clip(int(brain.pos_z / VOXEL_RES) + CENTER, 2, GRID_SIZE - 3)
        cv2.circle(floor_vis, (rx, rz), 4, (0, 255, 255), -1)
        
        yaw_rad = np.radians(-brain.yaw)
        hx = int(rx + 12 * np.sin(yaw_rad))
        hz = int(rz - 12 * np.cos(yaw_rad))
        cv2.line(floor_vis, (rx, rz), (hx, hz), (255, 255, 0), 2)
        
        surf = pygame.transform.scale(
            pygame.surfarray.make_surface(np.transpose(floor_vis, (1, 0, 2))), (280, 280))
        self.screen.blit(surf, (500, 490))
        self.screen.blit(self.font_sm.render("Floor Map + Person Path", True, (180, 180, 180)), (500, 775))
        
        # ===== RIGHT: Status Panel =====
        col_x = 800
        y = 10
        
        self.screen.blit(self.font_lg.render("TELLO V18.1", True, (100, 200, 255)), (col_x, y))
        y += 25
        self.screen.blit(self.font_sm.render("Person-to-Goal Path", True, (150, 150, 150)), (col_x, y))
        y += 25
        
        # Mode
        mode_col = (0, 255, 0) if auto else (100, 100, 255)
        self.screen.blit(self.font.render(f"{'AUTO' if auto else 'MANUAL'}", True, mode_col), (col_x, y))
        y += 30
        
        # State
        state_col = (255, 0, 0) if "EMERGENCY" in brain.mode else \
                    (255, 128, 0) if "ROTATING" in brain.mode or "BLOCKED" in brain.mode else \
                    (0, 255, 0) if "TARGET" in brain.mode else (200, 200, 200)
        self.screen.blit(self.font.render(brain.mode, True, state_col), (col_x, y))
        y += 30
        
        # ===== PATHFINDING STATUS SECTION =====
        self.screen.blit(self.font_sm.render("─── Pathfinding Status ───", True, (100, 100, 100)), (col_x, y))
        y += 18
        
        status_lines = brain.path_status.get_status_lines()
        for line in status_lines:
            # Color code based on content
            if "NOT FOUND" in line or "NOT SET" in line or "NO PATH" in line:
                color = (255, 100, 100)
            elif "DETECTED" in line or "waypoints" in line:
                color = (100, 255, 100)
            elif "⚠️" in line:
                color = (255, 200, 100)
            else:
                color = (180, 180, 180)
            
            self.screen.blit(self.font_xs.render(line, True, color), (col_x, y))
            y += 16
        
        y += 10
        
        # ===== Column depths bar graph =====
        self.screen.blit(self.font_sm.render("─── Column Depths ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        
        bar_width = 25
        bar_max_height = 60
        for i, depth in enumerate(column_info.depths):
            # Normalize depth to bar height (0-4m range)
            bar_height = int(min(depth / 4.0, 1.0) * bar_max_height)
            
            # Color based on depth
            if depth < EMERGENCY_DIST:
                color = (255, 0, 0)
            elif depth < CRITICAL_DIST:
                color = (255, 128, 0)
            elif depth < DANGER_DIST:
                color = (255, 255, 0)
            elif depth < CAUTION_DIST:
                color = (128, 255, 0)
            else:
                color = (0, 255, 0)
            
            bar_x = col_x + i * (bar_width + 2)
            bar_y = y + bar_max_height - bar_height
            
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, y, bar_width, bar_max_height), 1)
            
            # Mark best column
            if i == column_info.max_column:
                pygame.draw.rect(self.screen, (0, 255, 255), (bar_x - 2, y - 5, bar_width + 4, bar_max_height + 10), 2)
        
        y += bar_max_height + 15
        
        # Navigation info
        self.screen.blit(self.font_sm.render("─── Navigation ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        
        self.screen.blit(self.font_sm.render(
            f"Best: Col {column_info.max_column} @ {column_info.max_depth:.2f}m", 
            True, (0, 255, 255)), (col_x, y))
        y += 18
        self.screen.blit(self.font_sm.render(
            f"Center: {column_info.center_depth:.2f}m", True, (180, 180, 180)), (col_x, y))
        y += 18
        self.screen.blit(self.font_sm.render(
            f"Direction: {column_info.best_direction.upper()}", True, (0, 255, 0)), (col_x, y))
        y += 25
        
        # Speed indicator
        speed_text = f"SPEED: {column_info.recommended_speed}"
        speed_col = (255, 0, 0) if column_info.recommended_speed == 0 else \
                    (255, 255, 0) if column_info.recommended_speed < SPEED_CRUISE else (0, 255, 0)
        self.screen.blit(self.font.render(speed_text, True, speed_col), (col_x, y))
        y += 25
        
        # Position
        self.screen.blit(self.font_sm.render("─── Drone Position ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        self.screen.blit(self.font_sm.render(
            f"X: {brain.pos_x:.2f}  Y: {brain.pos_y:.2f}  Z: {brain.pos_z:.2f}",
            True, (180, 180, 180)), (col_x, y))
        y += 18
        self.screen.blit(self.font_sm.render(
            f"Yaw: {brain.yaw:.0f}°  Alt: {brain.current_altitude:.2f}m", 
            True, (180, 180, 180)), (col_x, y))
        y += 25
        
        # Controls
        self.screen.blit(self.font_sm.render("─── RC Commands ───", True, (100, 100, 100)), (col_x, y))
        y += 20
        fwd_color = (100, 150, 255) if rc[1] >= 0 else (255, 100, 100)
        self.screen.blit(self.font_sm.render(
            f"L={rc[0]:+3d} F={rc[1]:+3d} V={rc[2]:+3d} Y={rc[3]:+3d}",
            True, fwd_color), (col_x, y))
        y += 25
        
        # Battery
        batt_col = (0, 255, 0) if battery > 30 else (255, 165, 0) if battery > 15 else (255, 0, 0)
        self.screen.blit(self.font.render(f"BATTERY: {battery}%", True, batt_col), (col_x, y))
        y += 25
        
        # FPS
        fps_col = (0, 255, 0) if cam_fps > 25 else (255, 255, 0) if cam_fps > 15 else (255, 0, 0)
        self.screen.blit(self.font_sm.render(f"Video FPS: {cam_fps}", True, fps_col), (col_x, y))
        y += 18
        
        # GPU info
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.screen.blit(self.font_sm.render(
                f"GPU: {mem_used:.1f}/{mem_total:.1f}GB", True, (100, 255, 100)), (col_x, y))
        
        # ===== BOTTOM: Instructions =====
        y = 800
        instructions = [
            "[SPACE] Takeoff/Land  [M] Auto/Manual  [G] Set Goal  [T] Toggle Path Vis  [ESC] Quit",
            "[WASD] Move  [Arrows] Vert/Yaw  [SHIFT] Fast",
            f"Detecting: PERSON + {brain.target_obj} | V18.1: PERSON-TO-GOAL PATH PLANNING"
        ]
        for txt in instructions:
            self.screen.blit(self.font_sm.render(txt, True, (100, 100, 100)), (10, y))
            y += 18
        
        # Distance thresholds reference
        thresh_text = f"Thresholds: EMERGENCY<{EMERGENCY_DIST}m | CRITICAL<{CRITICAL_DIST}m | DANGER<{DANGER_DIST}m | SAFE>{SAFE_DIST}m"
        self.screen.blit(self.font_sm.render(thresh_text, True, (80, 80, 80)), (10, y + 10))
        
        pygame.display.flip()
        return self.clock.tick(30)


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("  TELLO V18.1: PERSON-TO-GOAL PATH PROJECTION")
    print("=" * 60)
    print()
    print("  KEY FEATURES:")
    print("  - PERSON DETECTION: Always-on YOLO World person detection")
    print("  - PERSON-TO-GOAL PATH: A* path from detected person to goal")
    print("  - PATH VISUALIZATION: Toggle with [T] key")
    print("  - PATHFINDING STATUS: Detailed logging in visualization")
    print("  - 3D POSITION ESTIMATION: Person world position from depth")
    print()
    print("  INHERITED FROM V18.0:")
    print("  - 12-COLUMN DEPTH ANALYSIS")
    print("  - DECISIVE OBSTACLE AVOIDANCE")
    print("  - EMERGENCY STOP at <0.5m")
    print("  - A* PATHFINDING with String Pulling + B-Spline")
    print()
    print("  CONTROLS:")
    print("  - [SPACE] Takeoff/Land")
    print("  - [M] Toggle Auto/Manual")
    print("  - [G] Set goal in current direction")
    print("  - [T] Toggle path visualization ON/OFF")
    print("  - [WASD] Move  [Arrows] Vert/Yaw")
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
    
    brain = DecisiveBrain(target)
    manual = ManualController()
    viz = Visualizer()
    
    brain.start_visualization()
    
    running = True
    auto_mode = False
    rc_cmd = (0, 0, 0, 0)
    column_info = ColumnDepthInfo()
    seg_vis = None
    path_overlay = None
    person_info = None
    
    print("[READY] Press SPACE to takeoff, M for auto mode, G to set goal, T to toggle path viz")
    
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
                    if event.key == pygame.K_g:
                        brain.set_goal_from_direction()
                    if event.key == pygame.K_t:
                        brain.toggle_path_visualization()
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
                depth_vis, column_vis, path_overlay, targets, rc_cmd, seg_vis, person_info = brain.get_action(frame)
                column_info = brain.column_analyzer.analyze(
                    brain.depth_engine.infer(frame)[1])
                
                # NO BACKWARD in auto mode
                if rc_cmd[1] < 0:
                    rc_cmd = (rc_cmd[0], 0, rc_cmd[2], rc_cmd[3])
                
                rc_cmd = tuple(int(v) for v in rc_cmd)
                drone.send_rc_control(*rc_cmd)
            else:
                rc_cmd = manual.get_rc(pygame.key.get_pressed())
                if drone.is_flying:
                    drone.send_rc_control(*rc_cmd)
                depth_vis, depth_m = brain.depth_engine.infer(frame)
                column_info = brain.column_analyzer.analyze(depth_m)
                column_vis = brain.column_analyzer.get_column_visualization(frame, depth_m, column_info)
                
                # Detect person and update path in manual mode too
                person_info_manual, _, boxes = brain._detect_objects(frame)
                brain.update_person_path(frame, depth_m, person_info_manual)
                person_info = person_info_manual
                
                # Update floor map
                brain.floor_map.update(depth_m, np.zeros(frame.shape[:2], dtype=np.uint8), 
                                       brain.pos_x, brain.pos_y, brain.pos_z, brain.yaw)
                
                path_overlay = brain.get_path_overlay(frame, person_info)
                
                if brain.floor_engine.enabled:
                    _, seg_vis = brain.floor_engine.segment_floor(frame)
                
                targets = boxes
            
            try:
                battery = drone.get_battery()
            except:
                battery = 0
            
            cam_fps = cam.get_fps()
            
            viz.render(frame, depth_vis, column_vis, path_overlay, brain, targets, auto_mode, 
                      battery, rc_cmd, column_info, seg_vis, cam_fps, person_info)
    
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