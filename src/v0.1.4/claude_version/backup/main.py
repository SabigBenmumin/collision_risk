import cv2
from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
from collections import defaultdict, deque
from filemanage import select_video_file, get_runid, create_outputfolder, handle_risk_event, select_txt_file
import get_fps

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# ==================== Debug Config ====================
SHOW_FRAME_INFO = True  # แสดง frame number / เวลา บนวิดีโอ

# ==================== Risk Detection Config ====================
TTC_THRESHOLD     = 3.0   # วินาที — ถ้า TTC < ค่านี้ถือว่าเสี่ยง
PET_THRESHOLD     = 2.0   # วินาที — ถ้า PET < ค่านี้ถือว่าเสี่ยง (near-miss)
RISK_COOLDOWN_S   = 5.0   # วินาที — เว้นระยะก่อน log คู่เดิมซ้ำ (ป้องกัน log ระเบิด)


class CalibrationTool:
    def __init__(self):
        self.points = []
        self.frame = None
        self.original_frame = None
        self.polygon = None
        self.homography_matrix = None
    
    def load_points_from_txt(self, file_path):
        """โหลดจุดจากไฟล์ txt
        รูปแบบไฟล์: แต่ละบรรทัดมี x,y หรือ x y
        ตัวอย่าง:
        100,200
        300,250
        350,400
        120,450
        """
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
            print(f"✓ Loaded {len(points)} points from {file_path}")
            for idx, pt in enumerate(self.points):
                print(f"  Point {idx+1}: ({pt[0]}, {pt[1]})")
            
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def input_points_keyboard(self):
        """ให้ผู้ใช้พิมพ์พิกัดผ่าน keyboard"""
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
                    print(f"  ✓ Point {i+1}: ({x}, {y})")
                    break
                except ValueError:
                    print("  ✗ Invalid format. Please use: x,y or x y (e.g., 100,200 or 100 200)")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        self.points = points
        print(f"\n✓ All 4 points entered successfully!")
        return True
    
    def draw_points_on_frame(self):
        """วาดจุดและเส้นบนภาพ"""
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
            cv2.imshow("Calibration", self.frame)
    
    def calculate_distances(self):
        """คำนวณระยะทางในหน่วยพิกเซลของแต่ละด้าน"""
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
        """สร้าง homography matrix สำหรับแปลงพิกัดจากภาพเป็นพิกัดจริง"""
        if len(self.points) != 4 or len(real_distances) != 4:
            print("Need 4 points and 4 real distances")
            return None
        
        src_points = np.float32(self.points)
        
        width = real_distances[0]   # top (1-2)
        height = real_distances[1]  # right (2-3)
        
        dst_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        H, status = cv2.findHomography(src_points, dst_points)
        
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
        """calibrate one zone - return {name, points, polygon, homography_matrix}"""
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
                    print("Instructions:")
                    print("1. Click 4 points to create a rectangle on the surface with known distances")
                    print("2. Click in the following order: top left -> top right -> bottom right -> bottom left")
                    print("3. Press 'c' when all 4 points are selected")
                    print("4. Press 'r' to restart")
                    print("5. Press 'q' to complete the calibration\n")
                    
                    cv2.namedWindow("Calibration")
                    cv2.setMouseCallback("Calibration", self.mouse_callback)
                    cv2.imshow("Calibration", self.frame)
                    
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('r'):
                            self.points = []
                            self.polygon = None
                            self.homography_matrix = None
                            self.frame = self.original_frame.copy()
                            cv2.imshow("Calibration", self.frame)
                            print("\nReseted - Start selecting a new point")
                        elif key == ord('c') and len(self.points) == 4:
                            break
                    
                    cv2.destroyAllWindows()
                    break
                    
                elif choice == '2':
                    file_path = select_txt_file()
                    if self.load_points_from_txt(file_path):
                        self.draw_points_on_frame()
                        cv2.namedWindow("Calibration")
                        cv2.imshow("Calibration", self.frame)
                        print("\nPress any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        break
                    else:
                        print("Failed to load file. Please try again.")
                        
                elif choice == '3':
                    if self.input_points_keyboard():
                        self.draw_points_on_frame()
                        cv2.namedWindow("Calibration")
                        cv2.imshow("Calibration", self.frame)
                        print("\nPress any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
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
                            dist_input = input(f"{label} ({distances[i]:.2f} px) = ")
                            real_dist = float(dist_input)
                            real_distances.append(real_dist)
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
                print(f"\n✓ Success! Homography matrix created")
        
        cap.release()
        return {
            'name': zone_name,
            'points': self.points.copy(),
            'polygon': self.get_polygon(),
            'homography_matrix': self.homography_matrix
        }
        
    def run(self, video_path):
        """Calibrate multiple zones - returns list of zone dicts"""
        zones = []
        zone_index = 0
        while True:
            zone_name = input(f"\nEnter zone name (or 'done' to finish)[{zone_index}]: ").strip()

            if zone_name.lower() == 'done':
                break
            if not zone_name:
                print(f"set zone name as '{zone_index}'")
                zone_name = str(zone_index)

            zone_data = self.calibrate_a_zone(video_path, zone_name)

            if zone_data is not None:
                zones.append(zone_data)
                print(f"✓ Zone '{zone_name}' calibrated successfully!")
                zone_index += 1
            else:
                print(f"✗ Failed to calibrate zone '{zone_name}'")

        return zones


# ==============================================================
# Geometry / Physics Helpers
# ==============================================================

def transform_point(point, homography_matrix):
    """แปลงจุดจากพิกัดภาพเป็นพิกัดจริง (เมตร)"""
    if homography_matrix is None:
        return None
    pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, homography_matrix)
    return transformed[0][0]


def calculate_real_distance(p1, p2, homography_matrix):
    """คำนวณระยะทางจริงระหว่าง 2 จุด (เมตร)"""
    if homography_matrix is None:
        return None
    real_p1 = transform_point(p1, homography_matrix)
    real_p2 = transform_point(p2, homography_matrix)
    if real_p1 is None or real_p2 is None:
        return None
    return float(np.hypot(real_p2[0] - real_p1[0], real_p2[1] - real_p1[1]))


def is_point_in_polygon(point, polygon):
    """ตรวจสอบว่าจุดอยู่ภายในพื้นที่ polygon หรือไม่"""
    if polygon is None:
        return True
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0


def find_nearest_object(current_idx, points):
    current_point = points[current_idx]
    min_distance = float("inf")
    for idx, point in enumerate(points):
        if idx == current_idx:
            continue
        dx = point[0] - current_point[0]
        dy = point[1] - current_point[1]
        if dy > 0 and abs(dx) < 50:
            distance = np.hypot(dx, dy)
            if distance < min_distance:
                min_distance = distance
    return min_distance if min_distance != float("inf") else None


def estimate_direction(track_points):
    """คำนวณทิศทางการเคลื่อนที่จาก track points

    track_points เป็น deque ของ tuple (x, y, frame_count)
    คืนค่า (angle, (vx, vy, x0, y0)) หรือ (None, None)
    """
    if len(track_points) < 5:
        return None, None
    
    # รองรับทั้ง (x, y) และ (x, y, frame)
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
        dot_product = vx * overall_dx + vy * overall_dy
        if dot_product < 0:
            vx = -vx
            vy = -vy
    
    x0, y0 = float(pts[-1][0]), float(pts[-1][1])
    angle = np.degrees(np.arctan2(vy, vx))
    return float(angle), (vx, vy, x0, y0)


def project_future_point(point, direction_vector, speed_mps, n_seconds, homography_matrix, track_points):
    """คาดคะเนตำแหน่งในอนาคตของวัตถุ"""
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

    # ตรวจสอบทิศใน real-world space
    pts_xy = [(p[0], p[1]) for p in track_points]
    if len(pts_xy) >= 2:
        first_real = transform_point(pts_xy[0], homography_matrix)
        last_real  = transform_point(pts_xy[-1], homography_matrix)
        if first_real is not None and last_real is not None:
            dx = last_real[0] - first_real[0]
            dy = last_real[1] - first_real[1]
            displacement_norm = np.sqrt(dx**2 + dy**2)
            if displacement_norm > 0.1:
                if (real_vx * dx + real_vy * dy) < 0:
                    real_vx, real_vy = -real_vx, -real_vy

    real_distance = speed_mps * n_seconds
    future_real_x = real_current[0] + real_vx * real_distance
    future_real_y = real_current[1] + real_vy * real_distance

    H_inv = np.linalg.inv(homography_matrix)
    future_pt = np.array([[[future_real_x, future_real_y]]], dtype=np.float32)
    future_pixel = cv2.perspectiveTransform(future_pt, H_inv)

    result = (int(future_pixel[0][0][0]), int(future_pixel[0][0][1]))

    MAX_PIXEL_DISTANCE = 500
    dx = result[0] - point[0]
    dy = result[1] - point[1]
    if np.sqrt(dx**2 + dy**2) > MAX_PIXEL_DISTANCE:
        return None

    return result


def draw_direction_line(frame, point, direction_vector, track_points=None, speed_mps=None,
                        n_seconds=2.0, homography_matrix=None,
                        fallback_length=60, color=(0, 0, 255), thickness=2):
    """วาดเส้นทำนายทิศทาง"""
    if direction_vector is None:
        return

    if speed_mps is not None and homography_matrix is not None and speed_mps > 0 and track_points is not None:
        future_point = project_future_point(point, direction_vector, speed_mps, n_seconds, homography_matrix, track_points)
        if future_point is not None:
            cv2.line(frame, (int(point[0]), int(point[1])), future_point, color, thickness)
            return

    vx, vy, x0, y0 = direction_vector
    end_x = int(point[0] + vx * fallback_length)
    end_y = int(point[1] + vy * fallback_length)
    cv2.line(frame, (int(point[0]), int(point[1])), (end_x, end_y), color, thickness)


def is_near_frame_edge(point, frame_shape, margin=50):
    h, w = frame_shape[:2]
    x, y = point
    return x < margin or x > w - margin or y < margin or y > h - margin


def save_or_not() -> bool:
    while True:
        saveinput = input("Would you like to save a result?[y/N]")
        if saveinput.lower() in ['yes', 'y']:
            return True
        elif saveinput.lower() in ['no', 'n', '']:
            return False
        else:
            print("your input is invalid, please try again!")


# ==============================================================
# TTC Computation
# ==============================================================

def compute_ttc(point_a, point_b, dir_vec_a, dir_vec_b,
                speed_a_mps, speed_b_mps, homography_matrix):
    """
    คำนวณ Time to Collision (วินาที) ระหว่างวัตถุสองตัวใน real-world space

    แนวคิด:
    - แปลงตำแหน่งปัจจุบันทั้งคู่เป็น real-world coordinates
    - คำนวณ closing speed = projection ของ relative velocity ตามแนวเส้นระหว่างวัตถุ
    - TTC = distance / closing_speed

    Returns: float (วินาที) หรือ None ถ้าวัตถุวิ่งออกจากกัน / คำนวณไม่ได้
    """
    if homography_matrix is None or dir_vec_a is None or dir_vec_b is None:
        return None
    if speed_a_mps <= 0 and speed_b_mps <= 0:
        return None

    real_a = transform_point(point_a, homography_matrix)
    real_b = transform_point(point_b, homography_matrix)
    if real_a is None or real_b is None:
        return None

    dist = float(np.hypot(real_b[0] - real_a[0], real_b[1] - real_a[1]))
    if dist < 0.1:
        return 0.0  # ชนกันแล้ว

    # velocity vector ของแต่ละตัวใน pixel space → scale ด้วย speed
    vx_a, vy_a = dir_vec_a[0] * speed_a_mps, dir_vec_a[1] * speed_a_mps
    vx_b, vy_b = dir_vec_b[0] * speed_b_mps, dir_vec_b[1] * speed_b_mps

    # relative velocity (A relative to B)
    rel_vx = vx_a - vx_b
    rel_vy = vy_a - vy_b

    # unit vector จาก A → B
    dx = (real_b[0] - real_a[0]) / dist
    dy = (real_b[1] - real_a[1]) / dist

    # closing speed = projection ของ relative velocity ตามแนวระหว่างวัตถุ
    # (ถ้าติดลบ = วิ่งออกจากกัน)
    closing_speed = rel_vx * dx + rel_vy * dy

    if closing_speed <= 0:
        return None  # ไม่มีแนวโน้มชน

    return dist / closing_speed


# ==============================================================
# PET Computation (จาก trace path intersection)
# ==============================================================

def _segment_intersection(p1, p2, p3, p4):
    """
    หาจุดตัดของ segment (p1→p2) กับ (p3→p4)
    คืนค่า (t, u) ที่ 0≤t≤1 และ 0≤u≤1 หมายถึงตัดกัน
    คืน None ถ้าขนาน / ไม่ตัดกัน
    """
    d1x = p2[0] - p1[0]; d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]; d2y = p4[1] - p3[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-10:
        return None  # ขนาน
    dx = p3[0] - p1[0]; dy = p3[1] - p1[1]
    t = (dx * d2y - dy * d2x) / denom
    u = (dx * d1y - dy * d1x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None


def compute_pet(track_a, track_b, fps):
    """
    คำนวณ Post Encroachment Time (PET) จาก trace path ของวัตถุสองตัว

    Parameters
    ----------
    track_a, track_b : deque ของ tuple (x, y, frame_count)
    fps              : float — frame rate จริง

    แนวคิด:
    - หาจุดตัดของ trace path ทั้งสอง (conflict point)
    - คำนวณว่าแต่ละตัวผ่านจุดนั้นที่ frame เท่าไร
    - PET = |frame_A_at_conflict - frame_B_at_conflict| / fps

    Returns: float (วินาที) หรือ None ถ้าหา conflict point ไม่ได้
    """
    if len(track_a) < 2 or len(track_b) < 2:
        return None

    pts_a = list(track_a)  # list of (x, y, frame)
    pts_b = list(track_b)

    best_pet = None

    # วน segment ทุกคู่
    for i in range(len(pts_a) - 1):
        for j in range(len(pts_b) - 1):
            p1 = (pts_a[i][0],   pts_a[i][1])
            p2 = (pts_a[i+1][0], pts_a[i+1][1])
            p3 = (pts_b[j][0],   pts_b[j][1])
            p4 = (pts_b[j+1][0], pts_b[j+1][1])

            result = _segment_intersection(p1, p2, p3, p4)
            if result is None:
                continue

            t, u = result

            # interpolate frame ที่แต่ละตัวผ่านจุดตัด
            frame_a = pts_a[i][2] + t * (pts_a[i+1][2] - pts_a[i][2])
            frame_b = pts_b[j][2] + u * (pts_b[j+1][2] - pts_b[j][2])

            pet = abs(frame_a - frame_b) / fps

            # เก็บ PET ที่น้อยที่สุด (conflict point ที่เกิดขึ้นล่าสุด / ใกล้ที่สุด)
            if best_pet is None or pet < best_pet:
                best_pet = pet

    return best_pet


# ==============================================================
# Risk Event Cooldown
# ==============================================================

def _risk_pair_key(id_a, id_b):
    """สร้าง key สำหรับคู่ tracker โดยไม่สนลำดับ"""
    return frozenset({id_a, id_b})


def should_log_risk(pair_key, event_type, current_time_s,
                    last_risk_logged, cooldown_s=RISK_COOLDOWN_S):
    """
    ตรวจว่าควร log event นี้หรือไม่ (cooldown per pair per event_type)

    Returns True ถ้า cooldown ผ่านแล้ว (ควร log)
    """
    key = (pair_key, event_type)
    last_t = last_risk_logged.get(key, -np.inf)
    if current_time_s - last_t >= cooldown_s:
        last_risk_logged[key] = current_time_s
        return True
    return False


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    video_path = select_video_file()
    cap = cv2.VideoCapture(video_path)
    speed_memory    = defaultdict(lambda: deque(maxlen=5))

    # [CHANGE] direction_memory เก็บ (x, y, frame_count) แทน (x, y)
    # เพื่อให้ compute_pet() รู้ว่าแต่ละ point ผ่านที่ frame เท่าไร
    # maxlen เพิ่มเป็น 60 เพื่อให้ trace ยาวพอหา intersection
    direction_memory = defaultdict(lambda: deque(maxlen=60))

    model = YOLO(r"models/best.pt")

    REAL_FPS = get_fps.get_video_fps_cv2(video_path=video_path)
    PROCESS_EVERY_N_FRAME = 1
    print(f"fps: {REAL_FPS}")
    FPS = REAL_FPS / PROCESS_EVERY_N_FRAME
    frame_count = 0

    byte_track = sv.ByteTrack(frame_rate=int(FPS))
    trace_annotator = sv.TraceAnnotator(trace_length=int(FPS*2), thickness=2, position=sv.Position.BOTTOM_CENTER)

    RUNID = get_runid()
    print(RUNID)

    is_save = save_or_not()
    OUTPUTDIR = None
    if is_save:
        OUTPUTDIR = create_outputfolder(RUNID)
        print(f"output directory is created at {OUTPUTDIR}")

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

    object_in_zone  = defaultdict(bool)
    last_speed      = defaultdict(float)
    speed_history   = defaultdict(list)

    # [NEW] cooldown dict: key=(frozenset{id_a,id_b}, event_type) → timestamp (วินาที)
    last_risk_logged: dict = {}

    # [NEW] cache direction vector ของแต่ละ tracker (อัปเดตทุก frame ใน per-object loop)
    cached_dir_vec: dict = {}

    # [NEW] cache class name ของแต่ละ tracker
    cached_class_name: dict = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAME != 0:
            continue

        current_time_s = frame_count / REAL_FPS  # เวลาปัจจุบันของวิดีโอ (วินาที)

        # วาด zone overlay
        for zone_idx, zone in enumerate(ZONES):
            overlay = frame.copy()
            color = (0, 255, 255)
            cv2.polylines(overlay, [zone['polygon']], True, color, 2)
            cv2.fillPoly(overlay, [zone['polygon']], color)
            frame = cv2.addWeighted(overlay, 0.05, frame, 0.95, 0)
            cx = int(zone['polygon'][:, 0].mean())
            cy = int(zone['polygon'][:, 1].mean())
            cv2.putText(frame, zone['name'], (cx-40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        results = model.predict(source=frame, stream=True, conf=0.65)
        result = next(results)

        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(int)
        annotated_frame = trace_annotator.annotate(scene=result.plot(), detections=detections)

        for zone_idx, zone in enumerate(ZONES):
            color = (0, 255, 255)
            cv2.polylines(annotated_frame, [zone['polygon']], True, color, 2)

        # ============================================================
        # Per-object loop: speed, direction, annotation
        # ============================================================
        for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
            if tracker_id is None:
                continue

            # อัปเดต class name cache
            class_id = detections.class_id[i]
            cached_class_name[tracker_id] = result.names[class_id] if class_id < len(result.names) else "unknown"

            matched_zones = []
            for zone_idx, zone in enumerate(ZONES):
                if is_point_in_polygon(point, zone['polygon']):
                    matched_zones.append((zone_idx, zone))
            in_zone = len(matched_zones) > 0

            # [CHANGE] เก็บ (x, y, frame_count) แทน point เปล่าๆ
            direction_memory[tracker_id].append((int(point[0]), int(point[1]), frame_count))

            angle, dir_vec = estimate_direction(direction_memory[tracker_id])

            # [NEW] cache direction vector ไว้ใช้ใน risk detection loop
            if dir_vec is not None:
                cached_dir_vec[tracker_id] = dir_vec

            if dir_vec is not None and not is_near_frame_edge(point, annotated_frame.shape):
                current_speed_mps = last_speed[tracker_id] / 3.6
                use_homography = matched_zones[0][1]['homography_matrix'] if matched_zones else None
                draw_direction_line(
                    annotated_frame, point, dir_vec,
                    track_points=list(direction_memory[tracker_id]),
                    speed_mps=current_speed_mps,
                    n_seconds=1,
                    homography_matrix=use_homography,
                    color=(0, 0, 255), thickness=2
                )

            if in_zone:
                for zone_idx, zone in matched_zones:
                    key = (tracker_id, zone_idx)
                    speed_memory[key].append(point)
                    object_in_zone[key] = True

                    if len(speed_memory[key]) >= 2:
                        p1 = speed_memory[key][0]
                        p2 = speed_memory[key][-1]
                        real_distance = calculate_real_distance(p1, p2, zone['homography_matrix'])

                        if real_distance is not None and real_distance > 0:
                            num_frames = len(speed_memory[key]) - 1
                            time_elapsed_s = num_frames / FPS

                            if time_elapsed_s > 0:
                                speed_mps = real_distance / time_elapsed_s
                                speed_kmh = speed_mps * 3.6
                                speed_history[tracker_id].append(speed_kmh)

                                if len(speed_history[tracker_id]) >= 3:
                                    display_speed = np.median(speed_history[tracker_id])
                                else:
                                    display_speed = speed_kmh

                                last_speed[tracker_id] = display_speed

                                cv2.putText(annotated_frame, f"Speed: {display_speed:.1f} km/h",
                                            (point[0] - 50, point[1] - 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                for zone_idx in range(len(ZONES)):
                    key = (tracker_id, zone_idx)
                    if object_in_zone.get(key, False):
                        if len(speed_history[tracker_id]) > 0:
                            speeds = np.array(speed_history[tracker_id])
                            q1 = np.percentile(speeds, 25)
                            q3 = np.percentile(speeds, 75)
                            iqr = q3 - q1
                            filtered = speeds[(speeds >= q1 - 1.5*iqr) & (speeds <= q3 + 1.5*iqr)]
                            if len(filtered) > 0:
                                last_speed[tracker_id] = float(np.median(filtered))

                        speed_memory[key].clear()
                        speed_history[tracker_id].clear()
                        object_in_zone[key] = False

                if tracker_id in last_speed and last_speed[tracker_id] > 0:
                    cv2.putText(annotated_frame, f"Speed: {last_speed[tracker_id]:.1f} km/h",
                                (point[0] - 50, point[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        # ============================================================
        # [NEW] Risk Detection Loop — TTC + PET (per pair)
        # ============================================================
        tracker_ids = [tid for tid in detections.tracker_id if tid is not None]
        n_obj = len(tracker_ids)

        for ii in range(n_obj):
            for jj in range(ii + 1, n_obj):
                id_a = tracker_ids[ii]
                id_b = tracker_ids[jj]

                pt_a = points[ii]
                pt_b = points[jj]

                dir_a = cached_dir_vec.get(id_a)
                dir_b = cached_dir_vec.get(id_b)

                speed_a_mps = last_speed.get(id_a, 0.0) / 3.6
                speed_b_mps = last_speed.get(id_b, 0.0) / 3.6

                class_a = cached_class_name.get(id_a, "unknown")
                class_b = cached_class_name.get(id_b, "unknown")

                # หา homography จาก zone ที่ครอบคลุมวัตถุอย่างน้อยหนึ่งตัว
                H = None
                zone_name_str = None
                for zone_idx, zone in enumerate(ZONES):
                    if (is_point_in_polygon(pt_a, zone['polygon']) or
                            is_point_in_polygon(pt_b, zone['polygon'])):
                        H = zone['homography_matrix']
                        zone_name_str = zone['name']
                        break

                pair_key = _risk_pair_key(id_a, id_b)
                mid_x = int((pt_a[0] + pt_b[0]) / 2)
                mid_y = int((pt_a[1] + pt_b[1]) / 2)

                # ---- TTC Check ----
                ttc = compute_ttc(pt_a, pt_b, dir_a, dir_b, speed_a_mps, speed_b_mps, H)
                if ttc is not None and ttc < TTC_THRESHOLD:
                    # วาด warning บนหน้าจอ
                    cv2.line(annotated_frame, tuple(pt_a), tuple(pt_b), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"TTC {ttc:.1f}s",
                                (mid_x - 35, mid_y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    # Log ถ้าผ่าน cooldown
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
                            metric_value=ttc,
                            zone_name=zone_name_str,
                            save_log=is_save,
                            save_frame=is_save,
                        )

                # ---- PET Check ----
                pet = compute_pet(direction_memory[id_a], direction_memory[id_b], REAL_FPS)
                if pet is not None and pet < PET_THRESHOLD:
                    # วาด warning บนหน้าจอ (สีต่างจาก TTC)
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
                            save_frame=is_save,
                        )

        # ============================================================
        # Frame info overlay
        # ============================================================
        if SHOW_FRAME_INFO:
            time_sec = frame_count / REAL_FPS
            minutes = int(time_sec // 60)
            seconds = time_sec % 60
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