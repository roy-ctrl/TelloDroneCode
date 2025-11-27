"""
Tello EDU: RTX 4060 "Metric Pseudo-LiDAR" + Visual Odometry
=============================================================
1. Fixed: 'numpy.ndarray' object has no attribute 'cpu' crash.
2. Fixed: Added missing Trajectory Plotting (Visual Odometry map).
3. Layout: Camera (Left), Depth (Top-Right), VO Map (Bottom-Right).
4. Fixed: VO Map Auto-scaling (prevents path from disappearing).
5. New: IMU Fusion (Stable Rotation) + 3D Mapping (Environment Memory).
"""

import os
import sys
import time
import threading
import logging
import cv2
import numpy as np
import pygame
import torch
import open3d as o3d
from djitellopy import Tello
from queue import Queue, Empty, Full
from dataclasses import dataclass
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
PROCESS_WIDTH = 480  
PROCESS_HEIGHT = 360
VO_SMOOTHING = 0.5 
MAPPING_ENABLED = True
VOXEL_SIZE = 0.05 # 5cm voxel grid for map memory

# Suppress Tello Logs
djitellopy_logger = logging.getLogger('djitellopy')
djitellopy_logger.setLevel(logging.WARN)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

@dataclass
class CameraIntrinsics:
    fx: float = 921.17
    fy: float = 919.02
    cx: float = 480.0
    cy: float = 360.0
    
    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

# ==========================================
# GPU ACCELERATED METRIC DEPTH (ZoeDepth)
# ==========================================
class ZoeDepthGPU:
    def __init__(self):
        print("\nInitializing ZoeDepth on RTX 4060 (FP32 Mode)...")
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✓ CUDA Detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("✗ WARNING: CUDA not detected. This will be slow.")

        try:
            # We use the torch hub to load the model
            repo = "isl-org/ZoeDepth"
            # Force reload to ensure consistency if cache is corrupted
            self.model = torch.hub.load(repo, "ZoeD_N", pretrained=True)
            self.model.to(self.device).eval()
            print("✓ ZoeDepth Model loaded successfully")
        except Exception as e:
            print(f"✗ Model load error: {e}")
            print("  TIP: Ensure you have internet for the first run.")
            self.model = None

    def estimate_metric_depth(self, frame: np.ndarray):
        if self.model is None: return None
        
        # Convert to PIL for the model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        with torch.no_grad():
            # Infer depth (returns meters)
            result = self.model.infer_pil(pil_image)
            
        # --- CRITICAL FIX START ---
        # The 'result' can be a numpy array OR a tensor depending on repo version.
        if isinstance(result, torch.Tensor):
            depth_map = result.detach().cpu().numpy()
        elif isinstance(result, np.ndarray):
            depth_map = result
        else:
            # Fallback for weird dictionary returns
            depth_map = np.array(result)
        # --- CRITICAL FIX END ---

        # Remove extra dimensions (1, 1, H, W) -> (H, W)
        return np.squeeze(depth_map)

# ==========================================
# THREADING INFRASTRUCTURE
# ==========================================
class AsyncProcessor:
    def __init__(self, name="Processor"):
        self.name = name
        self.input_queue = Queue(maxsize=1)
        self.output_lock = threading.Lock()
        self.latest_result = None
        self.running = False
        self.thread = None
        self.fps = 0.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)

    def update(self, frame, **kwargs):
        if not self.running: return
        try:
            try: self.input_queue.get_nowait()
            except Empty: pass
            self.input_queue.put_nowait((frame, kwargs))
        except Full: pass

    def get_result(self):
        with self.output_lock:
            return self.latest_result

    def _loop(self):
        while self.running:
            try:
                # Unpack tuple
                data = self.input_queue.get(timeout=0.1)
                frame, kwargs = data
                
                start_t = time.time()
                result = self.process(frame, **kwargs)
                with self.output_lock:
                    self.latest_result = result
                dt = time.time() - start_t
                if dt > 0: self.fps = 1.0 / dt
            except Empty: continue
            except Exception as e: print(f"{self.name} Error: {e}")

    def process(self, frame, **kwargs):
        raise NotImplementedError

class ThreadedDepth(AsyncProcessor):
    def __init__(self, estimator):
        super().__init__("DepthThread")
        self.estimator = estimator

    def process(self, frame, **kwargs):
        if self.estimator.model is None: return None

        # Resize for performance (ZoeDepth is heavy, even on RTX 4060)
        # 640x480 is good balance.
        small_frame = cv2.resize(frame, (640, 480))
        
        # Get Metric Depth
        depth_meters = self.estimator.estimate_metric_depth(small_frame)
        if depth_meters is None: return None

        # Double check dimensions
        if depth_meters.ndim > 2:
            depth_meters = np.squeeze(depth_meters)

        # Visualization (Heatmap)
        depth_clean = np.nan_to_num(depth_meters, posinf=0.0, neginf=0.0)
        
        # Color map for display
        depth_viz = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
        
        return {
            'depth_meters': depth_meters, 
            'viz': depth_color, 
            'rgb_frame': small_frame
        }

# ==========================================
# VISUAL ODOMETRY (Optical Flow + IMU)
# ==========================================
class ThreadedVO(AsyncProcessor):
    def __init__(self, camera: CameraIntrinsics):
        super().__init__("VOThread")
        self.camera = camera
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.prev_gray = None
        self.p0 = None
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.trajectory = [] # List of (x, z) coordinates
        self.scale = 1.0 
        self.t_smooth = np.zeros((3, 1))
        self.alpha = VO_SMOOTHING 

    def get_trajectory(self):
        with self.output_lock:
            return list(self.trajectory)
            
    def get_pose(self):
        with self.output_lock:
            return self.t_smooth.copy(), self.R.copy()

    def process(self, frame, **kwargs):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- IMU FUSION ---
        # Get the real yaw from the drone if provided. 
        # This replaces the unstable optical flow rotation estimation.
        imu_yaw = kwargs.get('yaw', None)
        if imu_yaw is not None:
            # Convert degrees to radians
            # Note: Tello yaw is usually -180 to 180. 
            # We assume 0 is start.
            rad = np.deg2rad(-imu_yaw) # Negative because Tello might be CW vs CCW specific
            
            # Create Rotation Matrix around Y axis (Gravity is Y in some CV coords, or Z)
            # In standard CV: X right, Y down, Z forward. 
            # Turning left/right is rotation around Y.
            self.R = np.array([
                [np.cos(rad), 0, np.sin(rad)],
                [0, 1, 0],
                [-np.sin(rad), 0, np.cos(rad)]
            ])
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return {'pos': self.t_smooth.flatten(), 'points': []}
        
        points_to_draw = []
        
        if self.p0 is not None and len(self.p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                
                # Check pixel displacement.
                if len(good_new) > 0:
                    flow_dist = np.mean(np.linalg.norm(good_new - good_old, axis=1))
                else:
                    flow_dist = 0
                
                if len(good_new) > 8 and flow_dist > 1.0:
                    try:
                        E, mask = cv2.findEssentialMat(good_new, good_old, self.camera.matrix, 
                                                     method=cv2.RANSAC, prob=0.999, threshold=1.0)
                        if E is not None:
                            _, R_est, t, _ = cv2.recoverPose(E, good_new, good_old, self.camera.matrix)
                            
                            # Only use Translation from Visual Odometry
                            # Use Rotation from IMU (self.R is already set above)
                            
                            # Apply rotation to translation vector to align with world
                            # World translation += Current Rotation * Local Translation
                            self.t = self.t + self.scale * (self.R @ t)
                            
                            # Smoothing
                            self.t_smooth = (self.alpha * self.t_smooth) + ((1 - self.alpha) * self.t)
                            
                            # Store trajectory 
                            x_pos = self.t_smooth[0][0]
                            z_pos = self.t_smooth[2][0]
                            
                            if not (np.isnan(x_pos) or np.isnan(z_pos)):
                                if len(self.trajectory) > 2000: self.trajectory.pop(0)
                                self.trajectory.append((x_pos, z_pos))
                            
                    except Exception: pass

                if len(good_new) < 100:
                    self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                else:
                    self.p0 = good_new.reshape(-1, 1, 2)
                
                points_to_draw = [(int(x), int(y)) for x, y in good_new]
            else:
                 self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        else:
             self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            
        self.prev_gray = gray.copy()
        return {'pos': self.t_smooth.flatten(), 'points': points_to_draw}

# ==========================================
# VIDEO GRABBER
# ==========================================
class VideoGrabber(threading.Thread):
    def __init__(self, udp_port=11111):
        super().__init__()
        self.udp_port = udp_port
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.cap = None

    def run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                url = f"udp://@0.0.0.0:{self.udp_port}"
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    time.sleep(0.5)
                    continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                self.cap.release()
                self.cap = None
                time.sleep(0.1)

    def start_capture(self):
        self.running = True
        self.start()

    def stop_capture(self):
        self.running = False
        if self.cap: self.cap.release()
        self.join(timeout=1.0)

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                # Tello camera is usually mirrored locally just like a webcam
                return cv2.flip(self.frame, 1) 
            return None

def draw_hud(surface, text, x, y, color=(255, 255, 255)):
    font = pygame.font.Font(None, 24)
    outline = font.render(text, True, (0,0,0))
    surf = font.render(text, True, color)
    surface.blit(outline, (x+1, y+1))
    surface.blit(surf, (x, y))

def draw_trajectory_map(surface, trajectory, width, height, color=(0, 255, 0)):
    """Draws the top-down trajectory of the drone with Auto-Scaling"""
    map_surf = pygame.Surface((width, height))
    map_surf.fill((20, 20, 40)) # Dark Blue background
    
    if len(trajectory) < 2:
        draw_hud(map_surf, "Waiting for motion...", 10, 10, (150, 255, 150))
        return map_surf

    traj_arr = np.array(trajectory)
    xs = traj_arr[:, 0]
    zs = traj_arr[:, 1]
    
    # Find bounding box
    min_x, max_x = np.min(xs), np.max(xs)
    min_z, max_z = np.min(zs), np.max(zs)
    range_x = max_x - min_x
    range_z = max_z - min_z
    
    padding = 40
    if range_x < 0.1: range_x = 1.0
    if range_z < 0.1: range_z = 1.0
    
    scale_x = (width - padding) / range_x
    scale_z = (height - padding) / range_z
    scale = min(scale_x, scale_z)
    scale = min(scale, 50.0)
    
    center_traj_x = (min_x + max_x) / 2
    center_traj_z = (min_z + max_z) / 2
    
    screen_cx = width // 2
    screen_cy = height // 2
    
    points = []
    for x, z in trajectory:
        px = screen_cx + int((x - center_traj_x) * scale)
        py = screen_cy - int((z - center_traj_z) * scale) 
        points.append((px, py))
        
    if len(points) > 1:
        pygame.draw.lines(map_surf, color, False, points, 2)
        
    if points:
        pygame.draw.circle(map_surf, (255, 0, 0), points[-1], 4)
        
    draw_hud(map_surf, f"VO (Scale: {scale:.1f}x)", 10, 10, (150, 255, 150))
    return map_surf

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    print("="*50)
    print("TELLO + ZOEDEPTH MAPPING (IMU FUSION)")
    print("="*50)

    # --- STEP 1: LOAD MODEL FIRST ---
    try:
        zoe_model = ZoeDepthGPU()
        camera = CameraIntrinsics()
    except Exception as e:
        print(f"CRITICAL: Model init failed. Error: {e}")
        return

    # --- STEP 2: CONNECT TO DRONE ---
    drone = Tello()
    try:
        drone.connect()
        drone.streamon() 
        print(f"✓ Connected. Battery: {drone.get_battery()}%")
    except Exception as e:
        print(f"(!) Drone connection failed: {e}")
        return

    # --- STEP 3: START THREADS ---
    grabber = VideoGrabber()
    grabber.start_capture()
    
    t_depth = ThreadedDepth(zoe_model)
    t_vo = ThreadedVO(camera)
    t_depth.start()
    t_vo.start()
    
    # Open3D Setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Environment Memory", width=640, height=480, left=50, top=50)
    
    # Global Map (The Memory)
    global_pcd = o3d.geometry.PointCloud()
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.4)
    ctr.rotate(0.0, -100.0)

    # Pygame Setup
    pygame.init()
    W, H = 960, 480 
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tello Cockpit - RTX 4060")
    
    running = True
    clock = pygame.time.Clock()
    
    print("✓ Systems Go. Press SPACE to take off.")
    
    frame_count = 0
    
    while running:
        raw_frame = grabber.get_frame()
        if raw_frame is None:
            vis.poll_events()
            vis.update_renderer()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            continue

        frame_count += 1
        
        # Resize for VO and Display
        frame_main = cv2.resize(raw_frame, (640, 480))
        frame_vo = cv2.resize(raw_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

        # Update Threads
        t_depth.update(raw_frame)
        
        # --- IMU FUSION UPDATE ---
        # Pass the IMU Yaw to the VO thread to stabilize rotation
        current_yaw = drone.get_yaw()
        t_vo.update(frame_vo, yaw=current_yaw)

        # --- OPEN3D UPDATE (MAPPING) ---
        depth_res = t_depth.get_result()
        if depth_res is not None:
            rgb_small = depth_res['rgb_frame'] # 640x480
            depth_meters = depth_res['depth_meters'] # 640x480
            
            # Create Local Point Cloud
            depth_mm = (depth_meters * 1000.0).astype(np.uint16)
            o3d_color = o3d.geometry.Image(cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(depth_mm)
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, 
                depth_scale=1000.0, 
                depth_trunc=10.0, # Limit range to 10m for cleaner map
                convert_rgb_to_intensity=False
            )
            
            pcd_local = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, 
                o3d.camera.PinholeCameraIntrinsic(640, 480, camera.fx, camera.fy, 320, 240)
            )
            
            # Get Current Pose from VO
            t_curr, R_curr = t_vo.get_pose()
            
            # Construct 4x4 Transformation Matrix
            # Open3D uses Y-down for camera, so we need to flip it to align with world
            # The order: Flip Camera -> Apply Rotation -> Apply Translation
            
            # 1. Flip Camera (CV to Open3D World)
            pcd_local.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            # 2. Apply Drone Pose
            # T_matrix = [ R  t ]
            #            [ 0  1 ]
            T_matrix = np.eye(4)
            T_matrix[:3, :3] = R_curr
            T_matrix[:3, 3] = t_curr.flatten()
            
            pcd_local.transform(T_matrix)
            
            # 3. Add to Global Map (Mapping / Memory)
            # Only add every 5th frame to save performance, or if moving
            if MAPPING_ENABLED and frame_count % 5 == 0:
                global_pcd += pcd_local
                
                # Voxel Downsample to keep memory usage low (The "Memory" part)
                # This merges overlapping points
                global_pcd = global_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
                
                vis.clear_geometries()
                vis.add_geometry(global_pcd)
            
            # Update View
            if not MAPPING_ENABLED:
                # Fallback to single frame mode if mapping disabled
                vis.clear_geometries()
                vis.add_geometry(pcd_local)
                
            vis.poll_events()
            vis.update_renderer()

        # --- PYGAME RENDER ---
        screen.fill((30, 30, 30))
        
        # 1. Main Camera Feed
        display_frame = frame_main.copy()
        vo_res = t_vo.get_result()
        if vo_res and 'points' in vo_res:
            for pt in vo_res['points']:
                x = int(pt[0] * (640/PROCESS_WIDTH))
                y = int(pt[1] * (480/PROCESS_HEIGHT))
                cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)

        frame_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
        frame_surf = pygame.transform.flip(frame_surf, False, False)
        screen.blit(frame_surf, (0, 0))

        # 2. Depth Heatmap
        if depth_res is not None:
            d_viz = cv2.resize(depth_res['viz'], (320, 240))
            d_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(d_viz, cv2.COLOR_BGR2RGB)))
            d_surf = pygame.transform.flip(d_surf, False, False)
            screen.blit(d_surf, (640, 0))
            draw_hud(screen, f"Points: {len(global_pcd.points)}", 650, 10, (255, 200, 100))
        else:
            draw_hud(screen, "Initializing Depth...", 650, 10, (255, 0, 0))

        # 3. Visual Odometry Map
        traj = t_vo.get_trajectory()
        vo_surf = draw_trajectory_map(screen, traj, 320, 240)
        screen.blit(vo_surf, (640, 240))
            
        bat = drone.get_battery()
        draw_hud(screen, f"Bat: {bat}%  Yaw: {current_yaw}", 10, 10)
        
        pygame.display.flip()

        # Input Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE:
                    if drone.is_flying: drone.land()
                    else: drone.takeoff()
                elif event.key == pygame.K_r:
                    # Reset Map
                    global_pcd = o3d.geometry.PointCloud()
                    t_vo.trajectory = []
                    t_vo.t = np.zeros((3,1))
                    t_vo.t_smooth = np.zeros((3,1))
                    print("Memory Reset.")

        keys = pygame.key.get_pressed()
        lr, fb, ud, yv = 0, 0, 0, 0
        speed = 50
        if keys[pygame.K_LEFT]: lr = -speed
        elif keys[pygame.K_RIGHT]: lr = speed
        if keys[pygame.K_UP]: fb = speed
        elif keys[pygame.K_DOWN]: fb = -speed
        if keys[pygame.K_w]: ud = speed
        elif keys[pygame.K_s]: ud = -speed
        if keys[pygame.K_a]: yv = -speed
        elif keys[pygame.K_d]: yv = speed
        
        if drone.is_flying:
            drone.send_rc_control(lr, fb, ud, yv)
            
        clock.tick(60)

    vis.destroy_window()
    grabber.stop_capture()
    t_depth.stop()
    t_vo.stop()
    drone.streamoff()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()