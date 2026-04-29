import cv2
from torch import classes
from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
from collections import defaultdict, deque
from filemanage import select_video_file, get_runid, create_outputfolder, handle_risk_event, select_txt_file, select_model_file, generate_summary_report
import get_fps

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# ==================== Debug Config ====================
SHOW_FRAME_INFO = True  # แสดง frame number / เวลา บนวิดีโอ

# ==================== Risk Detection Config ====================
TTC_THRESHOLD   = 2.0  # วินาที — ถ้า TTC_A หรือ TTC_B < ค่านี้ และ |TTC_A - TTC_B| < ARRIVAL_GAP ถือว่าเสี่ยง
ARRIVAL_GAP     = 1.5  # วินาที — ช่วงห่างสูงสุดของเวลาที่ทั้งคู่ถึง conflict point
PET_THRESHOLD   = 2.0  # วินาที — ถ้า PET < ค่านี้ถือว่าเสี่ยง (near-miss จาก trace จริง)
RISK_COOLDOWN_S = 5.0  # วินาที — เว้นระยะก่อน log คู่เดิมซ้ำ (ป้องกัน log ระเบิด)
TTC_LOOKAHEAD_S = 2.0  # วินาที — ความยาว future path ที่ใช้หา conflict point
LATERAL_OFFSET_MAX  = 2.0   # เมตร — ระยะห่างด้านข้างสูงสุดที่ถือว่า "เลนเดียวกัน" (ใช้กับ following / head-on)
DOT_FOLLOWING_MIN   = 0.7   # dot product ขั้นต่ำ สำหรับถือว่า "ทิศทางเดียวกัน" (following / rear-end)
DOT_HEADON_MAX      = -0.7  # dot product สูงสุด สำหรับถือว่า "สวนทาง" (head-on)

# ==================== Zone Scope Config ====================
# โหมดการตรวจจับความเสี่ยงเทียบกับ zone
#   "same_zone" = ทั้งคู่ต้องอยู่ zone เดียวกัน (เข้มงวด)
#   "any_zone"  = แค่ทั้งคู่อยู่ใน zone ใด zone หนึ่ง (แม้คนละ zone ก็ได้)
RISK_ZONE_MODE = "any_zone"

class CalibrationTool:
    def __init__(self):
        self.points = []
        self.frame = None
        self.original_frame = None
        self.polygon = None
        self.homography_matrix = None
    
    def load_points_from_txt(self, file_path):
        try:
            points = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ',' in line:
                        x, y = line.split(',')
                    else:
                        x, y = line.split()
                    points.append((int(float(x)), int(float(y))))
            
            if len(points) != 4:
                print(f"Error: txt file must contain exactly 4 points (found {len(points)})")
                return False
            
            self.points = points
            print(f"Loaded {len(points)} points from {file_path}")
            for idx, pt in enumerate(self.points):
                print(f"  Point {idx+1}: ({pt[0]}, {pt[1]})")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def input_points_keyboard(self):
        print("\n=== Input Coordinates via Keyboard ===")
        print("Enter 4 points (x,y) for:")
        print("  Point 1: Top-left")
        print("  Point 2: Top-right")
        print("  Point 3: Bottom-right")
        print("  Point 4: Bottom-left")
        
        points = []
        for i in range(4):
            while True:
                try:
                    coord_input = input(f"\nPoint {i+1} (x,y): ").strip()
                    if ',' in coord_input:
                        x, y = coord_input.split(',')
                    else:
                        x, y = coord_input.split()
                    x = int(float(x.strip()))
                    y = int(float(y.strip()))
                    points.append((x, y))
                    print(f"  Point {i+1}: ({x}, {y})")
                    break
                except ValueError:
                    print("  Invalid format. Please use: x,y or x y")
                except Exception as e:
                    print(f"  Error: {e}")
        
        self.points = points
        print(f"\nAll 4 points entered successfully!")
        return True
    
    def draw_points_on_frame(self):
        if self.original_frame is None:
            return
        self.frame = self.original_frame.copy()
        for idx, pt in enumerate(self.points):
            cv2.circle(self.frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(self.frame, str(idx+1), (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(self.points) > 1:
            for i in range(len(self.points)-1):
                cv2.line(self.frame, self.points[i], self.points[i+1], (255, 0, 0), 2)
        if len(self.points) == 4:
            cv2.line(self.frame, self.points[3], self.points[0], (255, 0, 0), 2)
            self.polygon = np.array(self.points, dtype=np.int32)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            self.draw_points_on_frame()
            win_name = param if param else "Calibration"
            cv2.imshow(win_name, self.frame)
    
    def calculate_distances(self):
        if len(self.points) != 4:
            print("There must be only 4 points!")
            return None
        distances = []
        labels = ["top (1-2)", "right (2-3)", "bottom (3-4)", "left (4-1)"]
        for i in range(4):
            p1 = self.points[i]
            p2 = self.points[(i+1) % 4]
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            distances.append(dist)
            print(f"{labels[i]}: {dist:.2f} pixel")
        return distances
    
    def create_homography(self, real_distances):
        if len(self.points) != 4 or len(real_distances) != 4:
            print("Need 4 points and 4 real distances")
            return None
        src_points = np.float32(self.points)
        width  = real_distances[0]
        height = real_distances[1]
        dst_points = np.float32([
            [0, 0], [width, 0], [width, height], [0, height]
        ])
        H, _ = cv2.findHomography(src_points, dst_points)
        print("\n=== Homography Matrix ===")
        print(H)
        print(f"Real-world area: {width:.2f}m x {height:.2f}m")
        return H
    
    def get_polygon(self):
        if len(self.points) == 4:
            return np.array(self.points, dtype=np.int32)
        return None
    
    def get_homography(self):
        return self.homography_matrix
    
    def calibrate_a_zone(self, video_path, zone_name):
        # Reset state จาก zone ก่อนหน้า
        self.points = []
        self.polygon = None
        self.homography_matrix = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: failed to open video")
            return None
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read a first frame")
            cap.release()
            return None

        self.original_frame = frame.copy()
        self.frame = frame.copy()

        print("\n=== Calibration Tool ===")
        print("Choose input method:")
        print("  1. Draw with mouse (interactive)")
        print("  2. Load from txt file")
        print("  3. Type coordinates via keyboard")
        
        while True:
            try:
                choice = input("\nEnter your choice (1/2/3): ").strip()
                if choice == '1':
                    print("\n=== Mouse Drawing Mode ===")
                    print("Click order: top-left -> top-right -> bottom-right -> bottom-left")
                    print("Press 'c' to confirm, 'r' to reset, 'q' to finish\n")
                    win_name = f"Calibration - {zone_name}"
                    cv2.namedWindow(win_name)
                    cv2.setMouseCallback(win_name, self.mouse_callback, param=win_name)
                    cv2.imshow(win_name, self.frame)
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('r'):
                            self.points = []
                            self.polygon = None
                            self.homography_matrix = None
                            self.frame = self.original_frame.copy()
                            cv2.imshow(win_name, self.frame)
                            print("\nReseted - Start selecting a new point")
                        elif key == ord('c') and len(self.points) == 4:
                            break
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)  # flush event queue
                    break
                elif choice == '2':
                    file_path = select_txt_file()
                    if self.load_points_from_txt(file_path):
                        self.draw_points_on_frame()
                        win_name = f"Calibration - {zone_name}"
                        cv2.namedWindow(win_name)
                        cv2.imshow(win_name, self.frame)
                        print("\nPress any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)  # flush event queue
                        break
                    else:
                        print("Failed to load file. Please try again.")
                elif choice == '3':
                    if self.input_points_keyboard():
                        self.draw_points_on_frame()
                        win_name = f"Calibration - {zone_name}"
                        cv2.namedWindow(win_name)
                        cv2.imshow(win_name, self.frame)
                        print("\nPress any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)  # flush event queue
                        break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except Exception as e:
                print(f"Error: {e}")
        
        if len(self.points) != 4:
            print("Error: Need exactly 4 points for calibration")
            cap.release()
            return None
        
        print("\n=== Distance as Pixels ===")
        distances = self.calculate_distances()
        
        if distances:
            print("\nPlease enter the actual distance (in meters) of each side:")
            print("NOTE: For homography, we mainly use top (width) and right (height)")
            while True:
                real_distances = []
                labels = ["top (1-2)", "right (2-3)", "bottom (3-4)", "left (4-1)"]
                for i, label in enumerate(labels):
                    while True:
                        try:
                            real_distances.append(float(input(f"{label} ({distances[i]:.2f} px) = ")))
                            break
                        except ValueError:
                            print("Please enter a number!")
                confirm = input("confirm? [Y/n]")
                if confirm.lower() == 'y' or confirm == '':
                    break
                else:
                    print("please input real distance again")

            self.homography_matrix = self.create_homography(real_distances)
            if self.homography_matrix is not None:
                print(f"\nSuccess! Homography matrix created")
        
        cap.release()
        return {
            'name': zone_name,
            'points': self.points.copy(),
            'polygon': self.get_polygon(),
            'homography_matrix': self.homography_matrix
        }
        
    def run(self, video_path):
        zones = []
        zone_index = 0
        while True:
            zone_name = input(f"\nEnter zone name (or 'done' to finish)[{zone_index}]: ").strip()
            if zone_name.lower() == 'done':
                break
            if not zone_name:
                zone_name = str(zone_index)
            zone_data = self.calibrate_a_zone(video_path, zone_name)
            if zone_data is not None:
                zones.append(zone_data)
                print(f"Zone '{zone_name}' calibrated successfully!")
                zone_index += 1
            else:
                print(f"Failed to calibrate zone '{zone_name}'")
        return zones

# ==============================================================
# Geometry Helpers
# ==============================================================

def transform_point(point, homography_matrix):
    """แปลงจุดจากพิกัดภาพเป็นพิกัดจริง (เมตร)"""
    if homography_matrix is None:
        return None
    pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, homography_matrix)[0][0]

def calculate_real_distance(p1, p2, homography_matrix):
    """คำนวณระยะทางจริงระหว่าง 2 จุด (เมตร)"""
    if homography_matrix is None:
        return None
    r1 = transform_point(p1, homography_matrix)
    r2 = transform_point(p2, homography_matrix)
    if r1 is None or r2 is None:
        return None
    return float(np.hypot(r2[0] - r1[0], r2[1] - r1[1]))

def is_point_in_polygon(point, polygon):
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def estimate_direction(track_points):
    """
    คำนวณทิศทางการเคลื่อนที่จาก track_points
    track_points: deque ของ (x, y, frame_count)
    คืนค่า (angle, (vx, vy, x0, y0)) หรือ (None, None)
    """
    if len(track_points) < 5:
        return None, None
    
    pts = [(p[0], p[1]) for p in track_points]
    n = len(pts)
    
    total_vx, total_vy, total_w = 0.0, 0.0, 0.0
    for i in range(1, n):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        w = i
        total_vx += dx * w
        total_vy += dy * w
        total_w += w
    
    if total_w == 0:
        return None, None
    
    vx = total_vx / total_w
    vy = total_vy / total_w
    norm = np.sqrt(vx**2 + vy**2)
    if norm < 1e-6:
        return None, None
    vx /= norm
    vy /= norm
    
    overall_dx = float(pts[-1][0] - pts[0][0])
    overall_dy = float(pts[-1][1] - pts[0][1])
    overall_norm = np.sqrt(overall_dx**2 + overall_dy**2)
    if overall_norm > 1e-6:
        if vx * overall_dx + vy * overall_dy < 0:
            vx = -vx
            vy = -vy
    
    x0, y0 = float(pts[-1][0]), float(pts[-1][1])
    return float(np.degrees(np.arctan2(vy, vx))), (vx, vy, x0, y0)

def project_future_point(point, direction_vector, speed_mps, n_seconds, homography_matrix, track_points):
    """คาดคะเนตำแหน่งในอนาคตของวัตถุ (pixel space) — ใช้วาด direction line บน frame"""
    if homography_matrix is None or direction_vector is None or speed_mps <= 0:
        return None

    vx, vy, x0, y0 = direction_vector
    real_current = transform_point(point, homography_matrix)
    if real_current is None:
        return None

    pt_base = transform_point((int(x0), int(y0)), homography_matrix)
    pt_tip  = transform_point((int(x0 + vx * 100), int(y0 + vy * 100)), homography_matrix)
    if pt_base is None or pt_tip is None:
        return None

    real_vx = pt_tip[0] - pt_base[0]
    real_vy = pt_tip[1] - pt_base[1]
    norm = np.sqrt(real_vx**2 + real_vy**2)
    if norm == 0:
        return None
    real_vx /= norm
    real_vy /= norm

    pts_xy = [(p[0], p[1]) for p in track_points]
    if len(pts_xy) >= 2:
        first_real = transform_point(pts_xy[0],  homography_matrix)
        last_real  = transform_point(pts_xy[-1], homography_matrix)
        if first_real is not None and last_real is not None:
            dx = last_real[0] - first_real[0]
            dy = last_real[1] - first_real[1]
            if np.sqrt(dx**2 + dy**2) > 0.1:
                if (real_vx * dx + real_vy * dy) < 0:
                    real_vx, real_vy = -real_vx, -real_vy

    future_real_x = real_current[0] + real_vx * speed_mps * n_seconds
    future_real_y = real_current[1] + real_vy * speed_mps * n_seconds

    H_inv = np.linalg.inv(homography_matrix)
    future_pixel = cv2.perspectiveTransform(
        np.array([[[future_real_x, future_real_y]]], dtype=np.float32), H_inv
    )
    result = (int(future_pixel[0][0][0]), int(future_pixel[0][0][1]))
    if np.hypot(result[0] - point[0], result[1] - point[1]) > 500:
        return None
    return result

def _get_real_direction_vector(direction_vector, homography_matrix, track_points):
    """
    แปลง direction vector จาก pixel space → real-world space
    พร้อม dot-product check เทียบกับ displacement จริง

    Returns: (real_vx, real_vy) หน่วยเวกเตอร์ใน real-world หรือ None
    """
    vx, vy, x0, y0 = direction_vector

    pt_base = transform_point((int(x0), int(y0)), homography_matrix)
    pt_tip  = transform_point((int(x0 + vx * 100), int(y0 + vy * 100)), homography_matrix)
    if pt_base is None or pt_tip is None:
        return None

    real_vx = pt_tip[0] - pt_base[0]
    real_vy = pt_tip[1] - pt_base[1]
    norm = np.sqrt(real_vx**2 + real_vy**2)
    if norm == 0:
        return None
    real_vx /= norm
    real_vy /= norm

    # dot-product check กับ displacement จริงจาก track
    pts_xy = [(p[0], p[1]) for p in track_points]
    if len(pts_xy) >= 2:
        first_real = transform_point(pts_xy[0],  homography_matrix)
        last_real  = transform_point(pts_xy[-1], homography_matrix)
        if first_real is not None and last_real is not None:
            dx = last_real[0] - first_real[0]
            dy = last_real[1] - first_real[1]
            if np.sqrt(dx**2 + dy**2) > 0.1:
                if (real_vx * dx + real_vy * dy) < 0:
                    real_vx, real_vy = -real_vx, -real_vy

    return real_vx, real_vy

def draw_direction_line(frame, point, direction_vector, track_points=None, speed_mps=None,
                        n_seconds=2.0, homography_matrix=None,
                        fallback_length=60, color=(0, 0, 255), thickness=2):
    if direction_vector is None:
        return
    if speed_mps is not None and homography_matrix is not None and speed_mps > 0 and track_points is not None:
        future_point = project_future_point(
            point, direction_vector, speed_mps, n_seconds, homography_matrix, track_points
        )
        if future_point is not None:
            cv2.line(frame, (int(point[0]), int(point[1])), future_point, color, thickness)
            return
    vx, vy, x0, y0 = direction_vector
    cv2.line(frame,
             (int(point[0]), int(point[1])),
             (int(point[0] + vx * fallback_length), int(point[1] + vy * fallback_length)),
             color, thickness)

def is_near_frame_edge(point, frame_shape, margin=50):
    h, w = frame_shape[:2]
    x, y = point
    return x < margin or x > w - margin or y < margin or y > h - margin

# TTC — Trajectory Intersection Method

def _ray_segment_intersection_2d(ox, oy, dx, dy, px, py, ex, ey):
    """
    หาจุดตัดของ ray (origin=(ox,oy), direction=(dx,dy))
    กับ segment (p→e) ใน 2D

    คืน (t_ray, u_seg):
      t_ray  ≥ 0        → จุดตัดอยู่บน ray (ไปข้างหน้า)
      0 ≤ u_seg ≤ 1    → จุดตัดอยู่บน segment
    หรือ None ถ้าไม่ตัด / ขนาน
    """
    sx = ex - px
    sy = ey - py
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-10:
        return None  # ขนาน
    t = ((px - ox) * sy - (py - oy) * sx) / denom
    u = ((px - ox) * dy - (py - oy) * dx) / denom
    if t >= 0 and 0.0 <= u <= 1.0:
        return t, u
    return None

def compute_ttc(
    point_a, point_b,
    dir_vec_a, dir_vec_b,
    speed_a_mps, speed_b_mps,
    homography_matrix,
    track_points_a, track_points_b,
    lookahead_s=TTC_LOOKAHEAD_S,
):
    """
    คำนวณ TTC โดยหา conflict point จาก trajectory จริง

    แนวคิด:
    ──────────────────────────────────────────────────────────────────
    ปัญหาของ TTC แบบเดิม (closing speed):
      → แค่วัดว่าวัตถุเคลื่อนเข้าหากันหรือเปล่า
      → วิ่งสวนทางกันคนละเลนก็ trigger ได้ เพราะ closing speed > 0
      → ไม่รู้ว่า path จะตัดกันจริงหรือเปล่า

    วิธีที่ถูกต้อง (trajectory intersection):
    1. Project future path ของแต่ละตัวออกไป lookahead_s วินาที
       ใน real-world space → segment A_now→A_future, B_now→B_future

    2. ถ้าสอง segment ไม่ตัดกัน → ไม่มี conflict point → return None
       (วิ่งสวนกันคนละเลน / ขนาน → ไม่เสี่ยง)

    3. ถ้าตัดกัน → conflict point C
         TTC_A = dist(A_now → C) / speed_A
         TTC_B = dist(B_now → C) / speed_B

    4. เสี่ยงเมื่อ:
         (TTC_A < TTC_THRESHOLD หรือ TTC_B < TTC_THRESHOLD)  ← ถึงเร็ว
         AND |TTC_A - TTC_B| < ARRIVAL_GAP                   ← ถึงพร้อมกัน

    Returns: (ttc_a, ttc_b) หรือ None ถ้าไม่มี conflict / ไม่เสี่ยง
    ──────────────────────────────────────────────────────────────────
    """
    if homography_matrix is None or dir_vec_a is None or dir_vec_b is None:
        return None
    if speed_a_mps <= 0 and speed_b_mps <= 0:
        return None

    # แปลงตำแหน่งปัจจุบัน → real-world
    real_a = transform_point(point_a, homography_matrix)
    real_b = transform_point(point_b, homography_matrix)
    if real_a is None or real_b is None:
        return None

    # แปลง direction vector → real-world space
    rdir_a = _get_real_direction_vector(dir_vec_a, homography_matrix, track_points_a)
    rdir_b = _get_real_direction_vector(dir_vec_b, homography_matrix, track_points_b)
    if rdir_a is None or rdir_b is None:
        return None

    rvx_a, rvy_a = rdir_a
    rvx_b, rvy_b = rdir_b

    # จุดปลายของ future path ใน real-world (lookahead_s วินาที)
    eff_speed_a = max(speed_a_mps, 0.5)
    eff_speed_b = max(speed_b_mps, 0.5)

    future_ax = real_a[0] + rvx_a * eff_speed_a * lookahead_s
    future_ay = real_a[1] + rvy_a * eff_speed_a * lookahead_s
    future_bx = real_b[0] + rvx_b * eff_speed_b * lookahead_s
    future_by = real_b[1] + rvy_b * eff_speed_b * lookahead_s

    # หา intersection: ray A ตัดกับ segment B_now→B_future
    result = _ray_segment_intersection_2d(
        real_a[0], real_a[1], rvx_a, rvy_a,   # ray A
        real_b[0], real_b[1], future_bx, future_by  # segment B
    )
    if result is None:
        return None  # path ไม่ตัดกัน → ไม่มี conflict point → ไม่เสี่ยง

    t_a, u_b = result

    # conflict point ใน real-world
    cx = real_a[0] + t_a * rvx_a
    cy = real_a[1] + t_a * rvy_a

    # ระยะจากตำแหน่งปัจจุบันถึง conflict point
    dist_a = float(np.hypot(cx - real_a[0], cy - real_a[1]))
    dist_b = float(np.hypot(cx - real_b[0], cy - real_b[1]))

    ttc_a = dist_a / eff_speed_a
    ttc_b = dist_b / eff_speed_b

    # เช็คเงื่อนไขความเสี่ยง
    arrive_soon     = (ttc_a < TTC_THRESHOLD or ttc_b < TTC_THRESHOLD)
    arrive_together = (abs(ttc_a - ttc_b) < ARRIVAL_GAP)

    if arrive_soon and arrive_together:
        return ttc_a, ttc_b

    return None

def compute_ttc_following(
    point_a, point_b,
    dir_vec_a, dir_vec_b,
    speed_a_mps, speed_b_mps,
    homography_matrix,
    track_points_a, track_points_b,
):
    """
    คำนวณ TTC สำหรับกรณีรถวิ่ง "ตามกัน" (rear-end / following)

    แนวคิด:
    ──────────────────────────────────────────────────────────────────
    เมื่อรถสองคันวิ่งทิศทางเดียวกัน (dot product > DOT_FOLLOWING_MIN):
    1. ใช้ทิศทางเฉลี่ยเป็นแกนอ้างอิง
    2. ฉาย separation vector ลงบนแกนนี้ → longitudinal gap
    3. คำนวณ lateral offset → ถ้าห่างเกิน LATERAL_OFFSET_MAX → คนละเลน → ไม่เสี่ยง
    4. closing speed = speed_behind − speed_ahead
       (ถ้า ≤ 0 → ไม่กำลังตามทัน → ไม่เสี่ยง)
    5. TTC = longitudinal_gap / closing_speed
    ──────────────────────────────────────────────────────────────────
    Returns: (ttc, ttc) หรือ None
    """
    if homography_matrix is None or dir_vec_a is None or dir_vec_b is None:
        return None
    if speed_a_mps <= 0 and speed_b_mps <= 0:
        return None

    real_a = transform_point(point_a, homography_matrix)
    real_b = transform_point(point_b, homography_matrix)
    if real_a is None or real_b is None:
        return None

    rdir_a = _get_real_direction_vector(dir_vec_a, homography_matrix, track_points_a)
    rdir_b = _get_real_direction_vector(dir_vec_b, homography_matrix, track_points_b)
    if rdir_a is None or rdir_b is None:
        return None

    rvx_a, rvy_a = rdir_a
    rvx_b, rvy_b = rdir_b

    # ตรวจว่าวิ่งทิศทางเดียวกัน
    dot = rvx_a * rvx_b + rvy_a * rvy_b
    if dot < DOT_FOLLOWING_MIN:
        return None

    # ใช้ทิศทางเฉลี่ยเป็นแกนอ้างอิง
    avg_dx = (rvx_a + rvx_b) / 2.0
    avg_dy = (rvy_a + rvy_b) / 2.0
    avg_norm = np.sqrt(avg_dx**2 + avg_dy**2)
    if avg_norm < 1e-6:
        return None
    avg_dx /= avg_norm
    avg_dy /= avg_norm

    # Separation vector A→B
    sep_x = real_b[0] - real_a[0]
    sep_y = real_b[1] - real_a[1]

    # Longitudinal (ตามแนวขับ) และ Lateral (ขวางเลน)
    longitudinal = sep_x * avg_dx + sep_y * avg_dy
    lateral = abs(sep_x * (-avg_dy) + sep_y * avg_dx)

    if lateral > LATERAL_OFFSET_MAX:
        return None  # คนละเลน

    gap = abs(longitudinal)
    if gap < 0.3:
        return None  # ใกล้เกินไป — ข้อมูลไม่น่าเชื่อถือ

    # หาว่าใครอยู่ข้างหลัง และคำนวณ closing speed
    if longitudinal > 0:
        # A อยู่ข้างหลัง B → A ต้องเร็วกว่าจึงจะชนตูด
        closing_speed = speed_a_mps - speed_b_mps
    else:
        # B อยู่ข้างหลัง A → B ต้องเร็วกว่า
        closing_speed = speed_b_mps - speed_a_mps

    if closing_speed <= 0.1:
        return None  # ไม่กำลังตามทัน หรือ closing ช้ามาก

    ttc = gap / closing_speed
    if ttc < TTC_THRESHOLD:
        return ttc, ttc

    return None

def compute_ttc_head_on(
    point_a, point_b,
    dir_vec_a, dir_vec_b,
    speed_a_mps, speed_b_mps,
    homography_matrix,
    track_points_a, track_points_b,
):
    """
    คำนวณ TTC สำหรับกรณีรถวิ่ง "สวนกัน" (head-on)

    แนวคิด:
    ──────────────────────────────────────────────────────────────────
    เมื่อรถสองคันวิ่งสวนทางกัน (dot product < DOT_HEADON_MAX):
    1. ใช้ทิศทางของ A เป็นแกนอ้างอิง
    2. ฉาย separation vector → longitudinal (ต้อง > 0 = B อยู่ข้างหน้า A)
    3. lateral offset → ถ้าห่างเกิน LATERAL_OFFSET_MAX → คนละเลน → ไม่เสี่ยง
    4. closing speed = speed_A + speed_B (มุ่งหน้าเข้าหากัน)
    5. TTC = longitudinal_gap / closing_speed
    ──────────────────────────────────────────────────────────────────
    Returns: (ttc, ttc) หรือ None
    """
    if homography_matrix is None or dir_vec_a is None or dir_vec_b is None:
        return None
    if speed_a_mps <= 0 and speed_b_mps <= 0:
        return None

    real_a = transform_point(point_a, homography_matrix)
    real_b = transform_point(point_b, homography_matrix)
    if real_a is None or real_b is None:
        return None

    rdir_a = _get_real_direction_vector(dir_vec_a, homography_matrix, track_points_a)
    rdir_b = _get_real_direction_vector(dir_vec_b, homography_matrix, track_points_b)
    if rdir_a is None or rdir_b is None:
        return None

    rvx_a, rvy_a = rdir_a
    rvx_b, rvy_b = rdir_b

    # ตรวจว่าวิ่งสวนทางกัน
    dot = rvx_a * rvx_b + rvy_a * rvy_b
    if dot > DOT_HEADON_MAX:
        return None

    # ใช้ทิศทาง A เป็นแกนอ้างอิง
    ref_dx, ref_dy = rvx_a, rvy_a

    # Separation vector A→B
    sep_x = real_b[0] - real_a[0]
    sep_y = real_b[1] - real_a[1]

    # Longitudinal (ตามแนว A) — B ต้องอยู่ข้างหน้า A จึงจะมุ่งเข้าหากัน
    longitudinal = sep_x * ref_dx + sep_y * ref_dy
    if longitudinal <= 0.3:
        return None  # ผ่านกันไปแล้ว หรือใกล้เกินไป

    # Lateral offset — ถ้าห่างทางด้านข้าง → คนละเลน → ไม่เสี่ยง
    lateral = abs(sep_x * (-ref_dy) + sep_y * ref_dx)
    if lateral > LATERAL_OFFSET_MAX:
        return None

    closing_speed = speed_a_mps + speed_b_mps
    if closing_speed <= 0.2:
        return None  # แทบไม่เคลื่อนที่

    ttc = longitudinal / closing_speed
    if ttc < TTC_THRESHOLD:
        return ttc, ttc

    return None

# PET — Trace Path Intersection

def _segment_intersection(p1, p2, p3, p4):
    """หาจุดตัดของ segment (p1→p2) กับ (p3→p4) คืน (t, u) หรือ None"""
    d1x = p2[0] - p1[0]; d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]; d2y = p4[1] - p3[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-10:
        return None
    dx = p3[0] - p1[0]; dy = p3[1] - p1[1]
    t = (dx * d2y - dy * d2x) / denom
    u = (dx * d1y - dy * d1x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None

def compute_pet(track_a, track_b, fps):
    """
    คำนวณ Post Encroachment Time (PET) จาก trace path จริง

    แนวคิด:
    ──────────────────────────────────────────────────────────────────
    ต่างจาก TTC ตรงที่:
      TTC → ดู future path (predictive) ก่อนเกิดเหตุ
      PET → ดู trace จริงที่เดินผ่านมาแล้ว (retrospective) หลัง near-miss

    วิธีคำนวณ:
    - วัตถุแต่ละตัวเก็บ trace ไว้เป็น (x, y, frame_count)
    - หา segment ใน trace A ที่ตัดกับ segment ใน trace B
      → จุดตัด = conflict point ที่เกิดขึ้นจริงในอดีต
    - interpolate หา frame ที่แต่ละตัวผ่านจุดนั้น
    - PET = |frame_A - frame_B| / fps
    - ถ้า PET น้อย → ทั้งคู่ผ่านจุดเดิมห่างกันน้อยมาก = near-miss จริง
    ──────────────────────────────────────────────────────────────────
    Returns: float (วินาที) หรือ None ถ้าไม่เคย cross กัน
    """
    if len(track_a) < 2 or len(track_b) < 2:
        return None

    pts_a = list(track_a)
    pts_b = list(track_b)
    best_pet = None

    for i in range(len(pts_a) - 1):
        for j in range(len(pts_b) - 1):
            p1 = (pts_a[i][0],   pts_a[i][1])
            p2 = (pts_a[i+1][0], pts_a[i+1][1])
            p3 = (pts_b[j][0],   pts_b[j][1])
            p4 = (pts_b[j+1][0], pts_b[j+1][1])

            res = _segment_intersection(p1, p2, p3, p4)
            if res is None:
                continue

            t, u = res
            frame_a = pts_a[i][2] + t * (pts_a[i+1][2] - pts_a[i][2])
            frame_b = pts_b[j][2] + u * (pts_b[j+1][2] - pts_b[j][2])
            pet = abs(frame_a - frame_b) / fps

            if best_pet is None or pet < best_pet:
                best_pet = pet

    return best_pet

# Risk Event Cooldown

def _risk_pair_key(id_a, id_b):
    return frozenset({id_a, id_b})


def should_log_risk(pair_key, event_type, current_time_s, last_risk_logged,
                    cooldown_s=RISK_COOLDOWN_S):
    """คืน True ถ้า cooldown ผ่านแล้ว และอัปเดต timestamp"""
    key = (pair_key, event_type)
    if current_time_s - last_risk_logged.get(key, -np.inf) >= cooldown_s:
        last_risk_logged[key] = current_time_s
        return True
    return False


# ==============================================================
# Session Config
# ==============================================================

def save_or_not() -> bool:
    while True:
        s = input("Would you like to save a result? [y/N] ").strip().lower()
        if s in ('y', 'yes'):
            return True
        elif s in ('n', 'no', ''):
            return False
        print("Invalid input, please try again!")


def capture_frame_or_not() -> bool:
    """ถามว่าจะ capture frame เป็นภาพเมื่อเกิด risk event หรือไม่"""
    while True:
        s = input("Capture frame as image when risk event occurs? [y/N] ").strip().lower()
        if s in ('y', 'yes'):
            return True
        elif s in ('n', 'no', ''):
            return False
        print("Invalid input, please try again!")


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    # model = YOLO(select_model_file())
    model = YOLO(r"models\\RT-DETR\\rtdetr-l.pt")
    video_path = select_video_file()
    cap = cv2.VideoCapture(video_path)
    classes_to_predict = [1, 2, 3, 5, 7]

    speed_memory     = defaultdict(lambda: deque(maxlen=5))
    # เก็บ (x, y, frame_count) — frame_count ใช้ interpolate เวลาสำหรับ PET
    direction_memory = defaultdict(lambda: deque(maxlen=60))


    REAL_FPS = get_fps.get_video_fps_cv2(video_path=video_path)
    PROCESS_EVERY_N_FRAME = 1
    print(f"fps: {REAL_FPS}")
    FPS = REAL_FPS / PROCESS_EVERY_N_FRAME
    frame_count = 0

    byte_track = sv.ByteTrack(frame_rate=int(FPS))
    trace_annotator = sv.TraceAnnotator(
        trace_length=int(FPS*2), thickness=2, position=sv.Position.BOTTOM_CENTER
    )

    RUNID = get_runid()
    print(RUNID)

    # --- Session config (ถามครั้งเดียวก่อนเริ่ม) ---
    is_save    = save_or_not()
    OUTPUTDIR  = None
    save_frame = False

    if is_save:
        OUTPUTDIR  = create_outputfolder(RUNID)
        print(f"Output directory: {OUTPUTDIR}")
        save_frame = capture_frame_or_not()  # ถามต่อเฉพาะเมื่อ save เปิดอยู่

    try:
        calibrator = CalibrationTool()
        ZONES = calibrator.run(video_path=video_path)
        if not ZONES:
            print("Error: No zones created. Exiting.")
            exit()
    except Exception as e:
        print(f"error: {e}")

    if not cap.isOpened():
        print("Error: could not open video file.")
        exit()

    object_in_zone    = defaultdict(bool)
    last_speed        = defaultdict(float)
    speed_history     = defaultdict(list)
    last_risk_logged: dict = {}
    cached_dir_vec:   dict = {}
    cached_class_name: dict = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAME != 0:
            continue

        current_time_s = frame_count / REAL_FPS

        # วาด zone overlay
        for zone in ZONES:
            overlay = frame.copy()
            color = (0, 255, 255)
            cv2.polylines(overlay, [zone['polygon']], True, color, 2)
            cv2.fillPoly(overlay, [zone['polygon']], color)
            frame = cv2.addWeighted(overlay, 0.05, frame, 0.95, 0)
            cx = int(zone['polygon'][:, 0].mean())
            cy = int(zone['polygon'][:, 1].mean())
            cv2.putText(frame, zone['name'], (cx-40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        results = model.predict(source=frame, stream=True, conf=0.65, classes=classes_to_predict)
        result  = next(results)

        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(int)
        annotated_frame = trace_annotator.annotate(scene=result.plot(), detections=detections)

        for zone in ZONES:
            cv2.polylines(annotated_frame, [zone['polygon']], True, (0, 255, 255), 2)

        # ============================================================
        # Per-object loop — speed, direction, annotation
        # ============================================================
        for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
            if tracker_id is None:
                continue

            class_id = detections.class_id[i]
            cached_class_name[tracker_id] = (
                result.names[class_id] if class_id < len(result.names) else "unknown"
            )

            matched_zones = [
                (zi, z) for zi, z in enumerate(ZONES)
                if is_point_in_polygon(point, z['polygon'])
            ]
            in_zone = len(matched_zones) > 0

            # บันทึก (x, y, frame_count) — สำคัญ: frame_count ใช้ interpolate เวลาสำหรับ PET
            direction_memory[tracker_id].append((int(point[0]), int(point[1]), frame_count))
            _, dir_vec = estimate_direction(direction_memory[tracker_id])
            if dir_vec is not None:
                cached_dir_vec[tracker_id] = dir_vec

            if dir_vec is not None and not is_near_frame_edge(point, annotated_frame.shape):
                use_h = matched_zones[0][1]['homography_matrix'] if matched_zones else None
                draw_direction_line(
                    annotated_frame, point, dir_vec,
                    track_points=list(direction_memory[tracker_id]),
                    speed_mps=last_speed[tracker_id] / 3.6,
                    n_seconds=1,
                    homography_matrix=use_h,
                    color=(0, 0, 255), thickness=2
                )

            if in_zone:
                for zone_idx, zone in matched_zones:
                    key = (tracker_id, zone_idx)
                    speed_memory[key].append(point)
                    object_in_zone[key] = True

                    if len(speed_memory[key]) >= 2:
                        real_dist = calculate_real_distance(
                            speed_memory[key][0], speed_memory[key][-1],
                            zone['homography_matrix']
                        )
                        if real_dist and real_dist > 0:
                            t_elapsed = (len(speed_memory[key]) - 1) / FPS
                            if t_elapsed > 0:
                                speed_kmh = (real_dist / t_elapsed) * 3.6
                                speed_history[tracker_id].append(speed_kmh)
                                display_speed = (
                                    float(np.median(speed_history[tracker_id]))
                                    if len(speed_history[tracker_id]) >= 3
                                    else speed_kmh
                                )
                                last_speed[tracker_id] = display_speed
                                cv2.putText(
                                    annotated_frame,
                                    f"Speed: {display_speed:.1f} km/h",
                                    (point[0] - 50, point[1] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                                )
            else:
                for zone_idx in range(len(ZONES)):
                    key = (tracker_id, zone_idx)
                    if object_in_zone.get(key, False):
                        speed_memory[key].clear()
                        speed_history[tracker_id].clear()
                        object_in_zone[key] = False

        # ============================================================
        # Risk Detection Loop — TTC + PET (per pair)
        # เงื่อนไข (ควบคุมโดย RISK_ZONE_MODE):
        #   "same_zone" → ทั้งคู่ต้องอยู่ zone เดียวกัน
        #   "any_zone"  → ทั้งคู่ต้องอยู่ใน zone ใด zone หนึ่ง (คนละ zone ได้)
        # ============================================================
        # กรองเฉพาะ tracker ที่มี id และอยู่ในอย่างน้อยหนึ่ง zone
        valid_indices = []
        object_zones: dict = {}  # tracker_id -> list ของ zone ที่อยู่
        for i, tid in enumerate(detections.tracker_id):
            if tid is None:
                continue
            zones_of_obj = [
                z for z in ZONES if is_point_in_polygon(points[i], z['polygon'])
            ]
            if zones_of_obj:
                valid_indices.append(i)
                object_zones[tid] = zones_of_obj

        n_obj = len(valid_indices)

        for ii in range(n_obj):
            for jj in range(ii + 1, n_obj):
                idx_a = valid_indices[ii]
                idx_b = valid_indices[jj]
                id_a = detections.tracker_id[idx_a]
                id_b = detections.tracker_id[idx_b]

                pt_a = points[idx_a]
                pt_b = points[idx_b]

                zones_a = object_zones.get(id_a, [])
                zones_b = object_zones.get(id_b, [])

                # หา shared zone (ถ้ามี) เพื่อใช้ homography ร่วมกัน
                shared_zones = [
                    za for za in zones_a
                    if any(zb['name'] == za['name'] for zb in zones_b)
                ]

                if RISK_ZONE_MODE == "same_zone":
                    # เข้มงวด: ต้องอยู่ zone เดียวกัน
                    if not shared_zones:
                        continue
                    chosen_zone = shared_zones[0]
                else:
                    # "any_zone" (default): ทั้งคู่อยู่ใน zone ใด zone หนึ่งก็พอ
                    # (valid_indices ได้การันตีแล้วว่าทั้งคู่อยู่ใน zone)
                    # เลือก homography จาก shared zone ก่อน ถ้าไม่มีก็ใช้ zone ของ A
                    if shared_zones:
                        chosen_zone = shared_zones[0]
                    else:
                        chosen_zone = zones_a[0]

                dir_a = cached_dir_vec.get(id_a)
                dir_b = cached_dir_vec.get(id_b)
                speed_a_mps = last_speed.get(id_a, 0.0) / 3.6
                speed_b_mps = last_speed.get(id_b, 0.0) / 3.6
                class_a = cached_class_name.get(id_a, "unknown")
                class_b = cached_class_name.get(id_b, "unknown")

                H = chosen_zone['homography_matrix']
                if shared_zones:
                    zone_name_str = chosen_zone['name']
                else:
                    # คนละ zone — แสดงทั้งคู่เพื่อให้ log ชัดเจน
                    zone_name_str = f"{zones_a[0]['name']}|{zones_b[0]['name']}"

                pair_key = _risk_pair_key(id_a, id_b)
                mid_x = int((pt_a[0] + pt_b[0]) / 2)
                mid_y = int((pt_a[1] + pt_b[1]) / 2)

                # ---- TTC Check (intersection / following / head-on) ----
                ttc_result = None
                collision_type = None
                _ttc_args = (
                    pt_a, pt_b, dir_a, dir_b,
                    speed_a_mps, speed_b_mps, H,
                    list(direction_memory[id_a]),
                    list(direction_memory[id_b]),
                )

                # 1) Intersection — path ตัดกัน
                ttc_result = compute_ttc(*_ttc_args)
                if ttc_result is not None:
                    collision_type = "intersection"

                # 2) Following — ตามกัน (rear-end)
                if ttc_result is None:
                    ttc_result = compute_ttc_following(*_ttc_args)
                    if ttc_result is not None:
                        collision_type = "following"

                # 3) Head-on — สวนกัน
                if ttc_result is None:
                    ttc_result = compute_ttc_head_on(*_ttc_args)
                    if ttc_result is not None:
                        collision_type = "head_on"

                if ttc_result is not None:
                    ttc_a, ttc_b = ttc_result
                    cv2.line(annotated_frame, tuple(pt_a), tuple(pt_b), (0, 0, 255), 2)
                    ttc_label = f"TTC {min(ttc_a, ttc_b):.1f}s"
                    if collision_type and collision_type != "intersection":
                        _type_short = {"following": "R", "head_on": "H"}
                        ttc_label += f" [{_type_short.get(collision_type, '')}]"
                    cv2.putText(annotated_frame, ttc_label,
                                (mid_x - 35, mid_y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    if should_log_risk(pair_key, "TTC_RISK", current_time_s, last_risk_logged):
                        handle_risk_event(
                            annotated_frame=annotated_frame,
                            output_dir=OUTPUTDIR,
                            frame_number=frame_count,
                            tracker_id_a=id_a,
                            tracker_id_b=id_b,
                            class_name_a=class_a,
                            class_name_b=class_b,
                            event_type="TTC_RISK",
                            metric_value=min(ttc_a, ttc_b),
                            zone_name=zone_name_str,
                            collision_type=collision_type,
                            save_log=is_save,
                            save_frame=save_frame,
                            video_fps=REAL_FPS,
                        )

                # ---- PET Check ----
                pet = compute_pet(direction_memory[id_a], direction_memory[id_b], REAL_FPS)
                if pet is not None and pet < PET_THRESHOLD:
                    cv2.line(annotated_frame, tuple(pt_a), tuple(pt_b), (0, 165, 255), 2)
                    cv2.putText(annotated_frame, f"PET {pet:.1f}s",
                                (mid_x - 35, mid_y + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

                    if should_log_risk(pair_key, "PET_RISK", current_time_s, last_risk_logged):
                        handle_risk_event(
                            annotated_frame=annotated_frame,
                            output_dir=OUTPUTDIR,
                            frame_number=frame_count,
                            tracker_id_a=id_a,
                            tracker_id_b=id_b,
                            class_name_a=class_a,
                            class_name_b=class_b,
                            event_type="PET_RISK",
                            metric_value=pet,
                            zone_name=zone_name_str,
                            save_log=is_save,
                            save_frame=save_frame,
                            video_fps=REAL_FPS,
                        )

        # Frame info overlay
        if SHOW_FRAME_INFO:
            time_sec  = frame_count / REAL_FPS
            minutes   = int(time_sec // 60)
            seconds   = time_sec % 60
            info_text = f"Frame: {frame_count}  Time: {minutes:02d}:{seconds:05.2f}"
            (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)
            cv2.putText(annotated_frame, info_text, (12, 12 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

    # สร้างสรุปการรัน
    if is_save and OUTPUTDIR:
        setup_config = {
            "REAL_FPS": REAL_FPS,
            "TTC_THRESHOLD": TTC_THRESHOLD,
            "ARRIVAL_GAP": ARRIVAL_GAP,
            "PET_THRESHOLD": PET_THRESHOLD,
            "RISK_COOLDOWN_S": RISK_COOLDOWN_S,
            "TTC_LOOKAHEAD_S": TTC_LOOKAHEAD_S,
            "LATERAL_OFFSET_MAX": LATERAL_OFFSET_MAX,
            "DOT_FOLLOWING_MIN": DOT_FOLLOWING_MIN,
            "DOT_HEADON_MAX": DOT_HEADON_MAX,
            "RISK_ZONE_MODE": RISK_ZONE_MODE,
        }
        input_info = {
            "video_path": video_path,
            "model_path": r"models\\RT-DETR\\rtdetr-l.pt",
            "num_zones": len(ZONES),
        }
        generate_summary_report(OUTPUTDIR, setup_config=setup_config, input_info=input_info)
