"""
Tello EDU: RTX 4060 Optimized "Metric Pseudo-LiDAR"
===================================================
Optimizations:
1. CUDA FP16 (Half-Precision) for ZoeDepth (uses Tensor Cores).
2. GPU-based Image Resizing/Normalization (removes CPU bottleneck).
3. Threaded Architecture: VO (CPU) and Depth (GPU) run in parallel.
4. Open3D Mapping + IMU Fusion.
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
import torch.nn.functional as F
import open3d as o3d
from djitellopy import Tello
from queue import Queue, Empty, Full
from dataclasses import dataclass

# ==========================================
# CONFIGURATION
# ==========================================
PROCESS_WIDTH = 480   
PROCESS_HEIGHT = 360
VO_SMOOTHING = 0.5 
MAPPING_ENABLED = True
VOXEL_SIZE = 0.05 

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

    def o3d_intrinsics(self):
        return o3d.camera.PinholeCameraIntrinsic(640, 480, self.fx, self.fy, 320, 240)

# ==========================================
# GPU ACCELERATED DEPTH (FP16 + Tensor Ops)
# ==========================================
class ZoeDepthRTX:
    def __init__(self):
        print("\nInitializing ZoeDepth on RTX 4060 (FP16 Tensor Core Mode)...")
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            props = torch.cuda.get_device_properties(0)
            print(f"✓ CUDA Detected: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
        else:
            print("✗ CRITICAL: CUDA not found. GPU acceleration disabled.")
            self.device = torch.device("cpu")

        try:
            # Load Model
            repo = "isl-org/ZoeDepth"
            self.model = torch.hub.load(repo, "ZoeD_N", pretrained=True)
            
            # OPTIMIZATION: Move to GPU and convert to Half Precision (FP16)
            self.model = self.model.to(self.device).eval()
            if self.device.type == 'cuda':
                self.model.half() # Uses RTX Tensor Cores
                print("✓ Model converted to FP16 (Half Precision)")
                
            print("✓ ZoeDepth Model loaded successfully")
        except Exception as e:
            print(f"✗ Model load error: {e}")
            self.model = None

    def infer(self, frame_tensor_cuda):
        """
        Input: CUDA Tensor (1, 3, H, W) normalized 0-1
        Output: Numpy Depth Map
        """
        if self.model is None: return None
        
        with torch.no_grad():
            # Ensure input is FP16 to match model
            if self.device.type == 'cuda':
                input_batch = frame_tensor_cuda.half()
            else:
                input_batch = frame_tensor_cuda

            # Infer
            depth_tensor = self.model.infer(input_batch)
            
            # Keep on GPU for as long as possible, only bring to CPU for viz/mapping final step
            if isinstance(depth_tensor, torch.Tensor):
                return depth_tensor.detach().cpu().numpy().squeeze()
            elif isinstance(depth_tensor, np.ndarray):
                return depth_tensor.squeeze()
            else:
                return np.array(depth_tensor).squeeze()

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

    def update(self, data, **kwargs):
        if not self.running: return
        try:
            try: self.input_queue.get_nowait()
            except Empty: pass
            self.input_queue.put_nowait((data, kwargs))
        except Full: pass

    def get_result(self):
        with self.output_lock:
            return self.latest_result

    def _loop(self):
        while self.running:
            try:
                data, kwargs = self.input_queue.get(timeout=0.1)
                start_t = time.time()
                result = self.process(data, **kwargs)
                with self.output_lock:
                    self.latest_result = result
                dt = time.time() - start_t
                if dt > 0: self.fps = 1.0 / dt
            except Empty: continue
            except Exception as e: print(f"{self.name} Error: {e}")

    def process(self, data, **kwargs):
        raise NotImplementedError

class ThreadedDepthGPU(AsyncProcessor):
    def __init__(self, estimator: ZoeDepthRTX):
        super().__init__("DepthThread")
        self.estimator = estimator
        self.device = estimator.device

    def process(self, frame_numpy, **kwargs):
        # OPTIMIZATION: Manual Tensor Preprocessing on GPU
        # 1. Convert Numpy (H,W,C) -> Tensor (C,H,W) -> Normalize 0-1
        frame_tensor = torch.from_numpy(frame_numpy).permute(2, 0, 1).float().div(255.0).to(self.device)
        
        # 2. Add batch dimension: (1, C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)
        
        # 3. Resize on GPU (Faster than cv2.resize on CPU)
        # ZoeDepth likes multiples of 32. 640x480 is standard.
        frame_resized = F.interpolate(frame_tensor, size=(480, 640), mode='bilinear', align_corners=False)

        # 4. Inference
        depth_meters = self.estimator.infer(frame_resized)
        if depth_meters is None: return None

        # 5. Prepare Visualization (CPU side)
        depth_clean = np.nan_to_num(depth_meters, posinf=0.0, neginf=0.0)
        depth_viz = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
        
        # Return the resized RGB frame (needed for mapping alignment)
        # We grab it back from the GPU tensor to ensure perfect alignment
        rgb_resized = frame_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() # (H,W,C)
        rgb_resized = (rgb_resized * 255).astype(np.uint8)

        return {
            'depth_meters': depth_meters, 
            'viz': depth_color, 
            'rgb_frame': rgb_resized
        }

# ==========================================
# VISUAL ODOMETRY (CPU - Cannot easily be GPU'd without heavy rewriting)
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
        self.trajectory = [] 
        self.t_smooth = np.zeros((3, 1))
        self.alpha = VO_SMOOTHING 
        self.scale = 1.0

    def get_trajectory(self):
        with self.output_lock:
            return list(self.trajectory)
            
    def get_pose(self):
        with self.output_lock:
            return self.t_smooth.copy(), self.R.copy()

    def process(self, frame, **kwargs):
        # Ensure grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # IMU Fusion
        imu_yaw = kwargs.get('yaw', None)
        if imu_yaw is not None:
            rad = np.deg2rad(-imu_yaw) 
            self.R = np.array([
                [np.cos(rad), 0, np.sin(rad)],
                [0, 1, 0],
                [-np.sin(rad), 0, np.cos(rad)]
            ])
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return {'points': []}
        
        points_to_draw = []
        if self.p0 is not None and len(self.p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                
                if len(good_new) > 8:
                    E, mask = cv2.findEssentialMat(good_new, good_old, self.camera.matrix, 
                                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    if E is not None:
                        _, R_est, t, _ = cv2.recoverPose(E, good_new, good_old, self.camera.matrix)
                        
                        # Translation logic
                        self.t = self.t + self.scale * (self.R @ t)
                        self.t_smooth = (self.alpha * self.t_smooth) + ((1 - self.alpha) * self.t)
                        
                        x_pos, z_pos = self.t_smooth[0][0], self.t_smooth[2][0]
                        if not (np.isnan(x_pos) or np.isnan(z_pos)):
                            if len(self.trajectory) > 2000: self.trajectory.pop(0)
                            self.trajectory.append((x_pos, z_pos))

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
        return {'points': points_to_draw}

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
                    self.frame = frame # Keep BGR
            else:
                self.cap.release()
                self.cap = None

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
                return cv2.flip(self.frame, 1) # Flip horizontally
            return None

# ==========================================
# UTILS (Rendering)
# ==========================================
def draw_hud(surface, text, x, y, color=(255, 255, 255)):
    font = pygame.font.Font(None, 24)
    outline = font.render(text, True, (0,0,0))
    surf = font.render(text, True, color)
    surface.blit(outline, (x+1, y+1))
    surface.blit(surf, (x, y))

def draw_trajectory_map(surface, trajectory, width, height):
    map_surf = pygame.Surface((width, height))
    map_surf.fill((20, 20, 40))
    
    if len(trajectory) < 2: return map_surf

    traj_arr = np.array(trajectory)
    xs, zs = traj_arr[:, 0], traj_arr[:, 1]
    
    # Auto-Scaling
    min_x, max_x = np.min(xs), np.max(xs)
    min_z, max_z = np.min(zs), np.max(zs)
    range_x, range_z = max_x - min_x, max_z - min_z
    
    scale = min((width-40)/max(range_x, 1.0), (height-40)/max(range_z, 1.0))
    scale = min(scale, 50.0) # Cap zoom
    
    cx, cz = (min_x + max_x) / 2, (min_z + max_z) / 2
    screen_cx, screen_cy = width // 2, height // 2
    
    points = []
    for x, z in trajectory:
        px = screen_cx + int((x - cx) * scale)
        py = screen_cy - int((z - cz) * scale) 
        points.append((px, py))
        
    if len(points) > 1: pygame.draw.lines(map_surf, (0, 255, 0), False, points, 2)
    if points: pygame.draw.circle(map_surf, (255, 0, 0), points[-1], 4)
    return map_surf

# ==========================================
# MAIN LOOP
# ==========================================
# ==========================================
# MAIN LOOP (UPDATED)
# ==========================================
def main():
    print("="*50)
    print("TELLO + ZOEDEPTH RTX 4060 OPTIMIZED")
    print("="*50)

    # 1. Initialize GPU Model
    zoe_model = ZoeDepthRTX()
    camera = CameraIntrinsics()

    # 2. Connect Drone
    drone = Tello()
    try:
        drone.connect()
        drone.streamon() 
        print(f"✓ Connected. Battery: {drone.get_battery()}%")
        # Optional: Force throw error if battery is too low for neural flight
        if drone.get_battery() < 20:
            print("(!) WARNING: Battery too low for reliable AI flight.")
    except Exception as e:
        print(f"(!) Connection failed: {e}")
        return

    # 3. Start Threads
    grabber = VideoGrabber()
    grabber.start_capture()
    
    t_depth = ThreadedDepthGPU(zoe_model)
    t_vo = ThreadedVO(camera)
    t_depth.start()
    t_vo.start()
    
    # 4. Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("RTX 4060 Memory Map", 640, 480, 50, 50)
    global_pcd = o3d.geometry.PointCloud()
    
    # 5. Pygame
    pygame.init()
    screen = pygame.display.set_mode((960, 480))
    pygame.display.set_caption("Tello Cockpit - CUDA 13.0")
    clock = pygame.time.Clock()
    
    print("✓ Ready. Press SPACE to Fly.")
    running = True
    frame_count = 0
    
    while running:
        # A. Grab latest frame (Non-blocking)
        raw_frame = grabber.get_frame()
        if raw_frame is None:
            vis.poll_events()
            vis.update_renderer()
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: running = False
            continue

        frame_count += 1
        
        # B. Send to threads
        frame_vo = cv2.resize(raw_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        t_depth.update(raw_frame) 
        t_vo.update(frame_vo, yaw=drone.get_yaw())

        # C. Retrieve Results
        depth_res = t_depth.get_result()
        vo_res = t_vo.get_result()

        # D. Mapping Logic (Point Cloud)
        if depth_res is not None:
            rgb_small = depth_res['rgb_frame'] 
            depth_meters = depth_res['depth_meters'] 
            
            # --- FIX 1: Sanitize Depth Data ---
            # Replace NaNs and Infs with 0.0 to prevent crash during cast
            depth_clean = np.nan_to_num(depth_meters, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create Open3D images
            depth_o3d = o3d.geometry.Image((depth_clean * 1000.0).astype(np.uint16))
            color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=8.0, convert_rgb_to_intensity=False)
            
            pcd_local = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, camera.o3d_intrinsics())
            
            # Transformation
            pcd_local.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            t_curr, R_curr = t_vo.get_pose()
            T_matrix = np.eye(4)
            T_matrix[:3, :3] = R_curr
            T_matrix[:3, 3] = t_curr.flatten()
            pcd_local.transform(T_matrix)
            
            if MAPPING_ENABLED and frame_count % 5 == 0:
                global_pcd += pcd_local
                if frame_count % 15 == 0: 
                    global_pcd = global_pcd.voxel_down_sample(VOXEL_SIZE)
                vis.clear_geometries()
                vis.add_geometry(global_pcd)
            elif not MAPPING_ENABLED:
                vis.clear_geometries()
                vis.add_geometry(pcd_local)

            vis.poll_events()
            vis.update_renderer()

        # E. Pygame Display
        screen.fill((30, 30, 30))
        
        disp_frame = frame_vo.copy()
        if vo_res and 'points' in vo_res:
            for pt in vo_res['points']:
                cv2.circle(disp_frame, pt, 2, (0, 255, 0), -1)
        
        frame_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)))
        frame_surf = pygame.transform.flip(frame_surf, False, False)
        screen.blit(pygame.transform.scale(frame_surf, (640, 480)), (0, 0))

        if depth_res is not None:
            d_viz = cv2.resize(depth_res['viz'], (320, 240))
            d_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(d_viz, cv2.COLOR_BGR2RGB)))
            d_surf = pygame.transform.flip(d_surf, False, False)
            screen.blit(d_surf, (640, 0))

        traj = t_vo.get_trajectory()
        vo_surf = draw_trajectory_map(screen, traj, 320, 240)
        screen.blit(vo_surf, (640, 240))
        
        bat = drone.get_battery()
        draw_hud(screen, f"Bat: {bat}%  FPS: {t_depth.fps:.1f}", 10, 10)
        
        pygame.display.flip()

        # Controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE:
                    # --- FIX 2: Safe Takeoff ---
                    if drone.is_flying: 
                        drone.land()
                    else: 
                        try:
                            drone.takeoff()
                        except Exception as e:
                            print(f"(!) Takeoff Refused: {e}")
                elif event.key == pygame.K_r:
                    global_pcd = o3d.geometry.PointCloud()
                    t_vo.trajectory = []
                    t_vo.t = np.zeros((3,1))
                    t_vo.t_smooth = np.zeros((3,1))

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