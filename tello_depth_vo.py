"""
Tello EDU Drone with Depth Anything V2 + OpenCV Visual Odometry
================================================================
FINAL WINDOWS FIX:
1. Bypasses 'djitellopy' video reader entirely (No 'av' library needed)
2. Uses direct OpenCV UDP capture
3. Threaded for high performance
"""

from djitellopy import Tello
import pygame
import cv2
import numpy as np
import time
import logging
import torch
from dataclasses import dataclass
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys
import os
import threading
from queue import Queue, Empty, Full

# Suppress logs
djitellopy_logger = logging.getLogger('djitellopy')
djitellopy_logger.setLevel(logging.WARN)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# ==========================================
# CONFIGURATION
# ==========================================
PROCESS_WIDTH = 480  
PROCESS_HEIGHT = 360
DEPTH_INPUT_SIZE = (320, 240) 

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

class DepthAnythingV2Local:
    def __init__(self, encoder: str = 'vitb', checkpoint_path: str = None):
        print("\nInitializing Depth Anything V2...")
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        if checkpoint_path is None:
            checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
            
        if not os.path.exists(checkpoint_path):
            print(f"✗ ERROR: Checkpoint not found: {checkpoint_path}")
            self.model = None
            return

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            config = self.model_configs[encoder]
            self.model = DepthAnythingV2(**config)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✓ Model loaded on CPU")
        except Exception as e:
            print(f"✗ Model load error: {e}")
            self.model = None

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None: return None
        try:
            with torch.no_grad():
                depth = self.model.infer_image(frame)
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_norm.astype(np.uint8)
            return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        except:
            return None

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

    def update(self, frame):
        if not self.running: return
        try:
            try: self.input_queue.get_nowait()
            except Empty: pass
            self.input_queue.put_nowait(frame)
        except Full: pass

    def get_result(self):
        with self.output_lock:
            return self.latest_result

    def _loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                start_t = time.time()
                result = self.process(frame)
                with self.output_lock:
                    self.latest_result = result
                dt = time.time() - start_t
                if dt > 0: self.fps = 1.0 / dt
            except Empty: continue
            except Exception as e: print(f"{self.name} Error: {e}")

    def process(self, frame):
        raise NotImplementedError

class ThreadedDepth(AsyncProcessor):
    def __init__(self, estimator):
        super().__init__("DepthThread")
        self.estimator = estimator

    def process(self, frame):
        small_frame = cv2.resize(frame, DEPTH_INPUT_SIZE)
        return self.estimator.estimate_depth(small_frame)

class ThreadedVO(AsyncProcessor):
    def __init__(self, camera: CameraIntrinsics):
        super().__init__("VOThread")
        self.camera = camera
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.trajectory = [self.t.flatten().copy()]
        self.prev_data = None 
        self.scale = 0.1

    def get_trajectory(self):
        with self.output_lock:
            return np.array(self.trajectory)

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if self.prev_data is None:
            self.prev_data = (gray, kp, des)
            return {'kp': kp, 'pos': self.t.flatten()}

        prev_gray, prev_kp, prev_des = self.prev_data

        if des is not None and prev_des is not None and len(kp) > 10 and len(prev_kp) > 10:
            try:
                matches = self.bf.knnMatch(prev_des, des, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                if len(good) >= 8:
                    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good])

                    E, mask = cv2.findEssentialMat(pts1, pts2, self.camera.matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    
                    if E is not None:
                        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera.matrix)
                        self.t = self.t + self.R @ (t * self.scale)
                        self.R = R @ self.R
                        if len(self.trajectory) > 5000: self.trajectory.pop(0)
                        self.trajectory.append(self.t.flatten().copy())

                self.prev_data = (gray, kp, des)
            except: pass
        
        return {'kp': kp, 'pos': self.t.flatten(), 'matches': len(good) if 'good' in locals() else 0}

class TrajectoryRenderer:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig, self.ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)

    def render(self, trajectory):
        self.ax.clear()
        if len(trajectory) > 1:
            x, z = trajectory[:, 0], trajectory[:, 2]
            self.ax.plot(x, z, 'b-', alpha=0.7)
            self.ax.plot(x[-1], z[-1], 'ro', markersize=5)
            mid_x, mid_z = np.mean(x), np.mean(z)
            span = max(np.ptp(x), np.ptp(z), 1.0) / 2 * 1.2
            self.ax.set_xlim(mid_x - span, mid_x + span)
            self.ax.set_ylim(mid_z - span, mid_z + span)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.canvas.draw()
        return cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

# ==========================================
# CUSTOM VIDEO GRABBER (Bypasses 'av' library)
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
                # Direct UDP connection string (Standard Tello Port)
                url = f"udp://@0.0.0.0:{self.udp_port}"
                
                # Try opening
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
        if self.cap:
            self.cap.release()
        self.join(timeout=1.0)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

def draw_hud(surface, text, x, y, color=(255, 255, 255)):
    font = pygame.font.Font(None, 24)
    outline = font.render(text, True, (0,0,0))
    surf = font.render(text, True, color)
    surface.blit(outline, (x+1, y+1))
    surface.blit(surf, (x, y))

def main():
    print("="*50)
    print("TELLO CONTROLLER (NO-AV VERSION)")
    print("="*50)

    # 1. Initialize Drone
    drone = Tello()
    try:
        drone.connect()
        drone.streamon() 
        print(f"✓ Connected. Battery: {drone.get_battery()}%")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 2. Start Video Grabber
    grabber = VideoGrabber()
    grabber.start_capture()
    print("✓ Video Capture Started")

    # 3. Initialize Models
    depth_model = DepthAnythingV2Local(encoder='vitb')
    camera = CameraIntrinsics()

    # 4. Threads
    t_depth = ThreadedDepth(depth_model)
    t_vo = ThreadedVO(camera)
    t_depth.start()
    t_vo.start()
    
    # 5. UI Setup
    pygame.init()
    W, H = 960, 480 
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tello Async Control")
    traj_viz = TrajectoryRenderer(320, 240)
    
    running = True
    clock = pygame.time.Clock()
    show_vo = True
    show_depth = True
    
    print("Waiting for video... (Ensure Windows Firewall allows Python)")
    
    while running:
        # --- Video ---
        raw_frame = grabber.get_frame()
        
        if raw_frame is None:
            screen.fill((0,0,0))
            draw_hud(screen, "Waiting for video stream...", W//2-100, H//2)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            continue

        # Resize
        frame_main = cv2.resize(raw_frame, (640, 480))
        frame_proc = cv2.resize(raw_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

        # --- Update Threads ---
        if show_depth: t_depth.update(frame_main)
        if show_vo: t_vo.update(frame_proc)

        # --- Draw ---
        screen.fill((30, 30, 30))
        
        # Video
        display_frame = frame_main.copy()
        vo_res = t_vo.get_result()
        if show_vo and vo_res and 'kp' in vo_res:
            for kp in vo_res['kp']:
                x = int(kp.pt[0] * (640/PROCESS_WIDTH))
                y = int(kp.pt[1] * (480/PROCESS_HEIGHT))
                cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)

        frame_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
        frame_surf = pygame.transform.flip(frame_surf, False, False)
        screen.blit(frame_surf, (0, 0))

        # Depth
        depth_res = t_depth.get_result()
        if show_depth and depth_res is not None:
            d_viz = cv2.resize(depth_res, (320, 240))
            d_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(d_viz, cv2.COLOR_BGR2RGB)))
            d_surf = pygame.transform.flip(d_surf, False, False)
            screen.blit(d_surf, (640, 0))

        # Trajectory
        if show_vo:
            traj_arr = t_vo.get_trajectory()
            if len(traj_arr) > 1:
                traj_img = traj_viz.render(traj_arr)
                t_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(traj_img, cv2.COLOR_BGR2RGB)))
                t_surf = pygame.transform.flip(t_surf, False, False)
                screen.blit(t_surf, (640, 240))
        
        # HUD
        bat = drone.get_battery()
        draw_hud(screen, f"Battery: {bat}%  FPS: {clock.get_fps():.0f}", 10, 10)
        draw_hud(screen, f"Depth FPS: {t_depth.fps:.1f}", 10, 30, (255, 200, 100))
        draw_hud(screen, f"VO FPS: {t_vo.fps:.1f}", 10, 50, (100, 255, 100))

        pygame.display.flip()

        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE:
                    if drone.is_flying: drone.land()
                    else: drone.takeoff()
                elif event.key == pygame.K_v: show_vo = not show_vo
                elif event.key == pygame.K_b: show_depth = not show_depth

        # RC
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

    # Cleanup
    print("Stopping...")
    grabber.stop_capture()
    t_depth.stop()
    t_vo.stop()
    drone.streamoff()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()