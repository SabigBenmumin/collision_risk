"""
Debug Logger for Collision Risk Detector v0.1.2
เครื่องมือ debug สำหรับวิเคราะห์ bug เส้นทำนายทิศทาง

ฟีเจอร์:
- ดึง class list จาก model แล้วให้ user เลือก class ที่ต้องการ
- กำหนด frame range หรือช่วงเวลาที่ต้องการวิเคราะห์
- เก็บ log ละเอียดของ direction vector, dot products, track points ทุก frame
- Export เป็น CSV + JSON สำหรับวิเคราะห์ bug
"""

import cv2
from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
import json
import csv
from collections import defaultdict, deque
from filemanage import select_video_file, get_runid, create_outputfolder, select_txt_file
import get_fps

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==================== CalibrationTool (เหมือน v0.1.1) ====================

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
            print(f"✓ Loaded {len(points)} points from {file_path}")
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
                    print(f"  ✓ Point {i+1}: ({x}, {y})")
                    break
                except ValueError:
                    print("  ✗ Invalid format. Please use: x,y or x y")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        self.points = points
        print(f"\n✓ All 4 points entered successfully!")
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
            cv2.imshow("Calibration", self.frame)

    def calculate_distances(self):
        if len(self.points) != 4:
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
            return None
        src_points = np.float32(self.points)
        width = real_distances[0]
        height = real_distances[1]
        dst_points = np.float32([
            [0, 0], [width, 0], [width, height], [0, height]
        ])
        H, status = cv2.findHomography(src_points, dst_points)
        print(f"\n=== Homography Matrix ===")
        print(H)
        print(f"Real-world area: {width:.2f}m x {height:.2f}m")
        return H

    def get_polygon(self):
        if len(self.points) == 4:
            return np.array(self.points, dtype=np.int32)
        return None

    def get_homography(self):
        return self.homography_matrix

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: failed to open video")
            return None, None
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, None
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
                    print("Click 4 points: top-left -> top-right -> bottom-right -> bottom-left")
                    print("Press 'c' to confirm, 'r' to reset, 'q' to quit\n")
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
                            self.frame = self.original_frame.copy()
                            cv2.imshow("Calibration", self.frame)
                        elif key == ord('c') and len(self.points) == 4:
                            break
                    cv2.destroyAllWindows()
                    break
                elif choice == '2':
                    file_path = select_txt_file()
                    if self.load_points_from_txt(file_path):
                        self.draw_points_on_frame()
                        cv2.imshow("Calibration", self.frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        break
                elif choice == '3':
                    if self.input_points_keyboard():
                        self.draw_points_on_frame()
                        cv2.imshow("Calibration", self.frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        break
                else:
                    print("Invalid choice.")
            except Exception as e:
                print(f"Error: {e}")

        if len(self.points) != 4:
            cap.release()
            return None, None

        print("\n=== Distance as Pixels ===")
        distances = self.calculate_distances()
        if distances:
            print("\nPlease enter the actual distance (in meters) of each side:")
            real_distances = []
            labels = ["top (1-2)", "right (2-3)", "bottom (3-4)", "left (4-1)"]
            for i, label in enumerate(labels):
                while True:
                    try:
                        real_dist = float(input(f"{label} ({distances[i]:.2f} px) = "))
                        real_distances.append(real_dist)
                        break
                    except ValueError:
                        print("Please enter a number!")
            self.homography_matrix = self.create_homography(real_distances)

        cap.release()
        return self.get_homography(), self.get_polygon()


# ==================== Utility Functions ====================

def transform_point(point, homography_matrix):
    if homography_matrix is None:
        return None
    pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, homography_matrix)
    return transformed[0][0]

def calculate_real_distance(p1, p2, homography_matrix):
    if homography_matrix is None:
        return None
    real_p1 = transform_point(p1, homography_matrix)
    real_p2 = transform_point(p2, homography_matrix)
    if real_p1 is None or real_p2 is None:
        return None
    return np.sqrt((real_p2[0] - real_p1[0])**2 + (real_p2[1] - real_p1[1])**2)

def is_point_in_polygon(point, polygon):
    if polygon is None:
        return True
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0

def is_near_frame_edge(point, frame_shape, margin=50):
    h, w = frame_shape[:2]
    x, y = point
    return x < margin or x > w - margin or y < margin or y > h - margin


# ==================== Direction Estimation (with diagnostic data) ====================

def estimate_direction_debug(track_points):
    """คำนวณทิศทาง พร้อมส่ง diagnostic data กลับมาด้วย"""
    diag = {
        "vx_raw": None, "vy_raw": None,
        "overall_dx": None, "overall_dy": None,
        "dot_product_image": None,
        "direction_flipped_image": False,
    }

    if len(track_points) < 5:
        return None, None, diag

    pts = list(track_points)
    n = len(pts)

    # Weighted average displacement
    total_vx, total_vy, total_w = 0.0, 0.0, 0.0
    for i in range(1, n):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        w = i
        total_vx += dx * w
        total_vy += dy * w
        total_w += w

    if total_w == 0:
        return None, None, diag

    vx = total_vx / total_w
    vy = total_vy / total_w

    norm = np.sqrt(vx**2 + vy**2)
    if norm < 1e-6:
        return None, None, diag
    vx /= norm
    vy /= norm

    diag["vx_raw"] = round(float(vx), 6)
    diag["vy_raw"] = round(float(vy), 6)

    # Dot product check กับ overall displacement
    overall_dx = float(pts[-1][0] - pts[0][0])
    overall_dy = float(pts[-1][1] - pts[0][1])
    overall_norm = np.sqrt(overall_dx**2 + overall_dy**2)

    diag["overall_dx"] = round(overall_dx, 2)
    diag["overall_dy"] = round(overall_dy, 2)

    if overall_norm > 1e-6:
        dot_product = vx * overall_dx + vy * overall_dy
        diag["dot_product_image"] = round(float(dot_product), 4)
        if dot_product < 0:
            vx = -vx
            vy = -vy
            diag["direction_flipped_image"] = True

    x0, y0 = float(pts[-1][0]), float(pts[-1][1])
    angle = np.degrees(np.arctan2(vy, vx))
    return float(angle), (vx, vy, x0, y0), diag


def project_future_point_debug(point, direction_vector, speed_mps, n_seconds, homography_matrix, track_points):
    """คาดคะเนตำแหน่งอนาคต พร้อม diagnostic data"""
    diag = {
        "real_vx": None, "real_vy": None,
        "dot_product_real": None,
        "direction_flipped_real": False,
        "future_point": None,
        "prediction_method": "none",
    }

    if homography_matrix is None or direction_vector is None or speed_mps <= 0:
        return None, diag

    vx, vy, x0, y0 = direction_vector
    real_current = transform_point(point, homography_matrix)
    if real_current is None:
        return None, diag

    pt_base = transform_point((int(x0), int(y0)), homography_matrix)
    pt_tip  = transform_point((int(x0 + vx * 100), int(y0 + vy * 100)), homography_matrix)
    if pt_base is None or pt_tip is None:
        return None, diag

    real_vx = pt_tip[0] - pt_base[0]
    real_vy = pt_tip[1] - pt_base[1]
    norm = np.sqrt(real_vx**2 + real_vy**2)
    if norm == 0:
        return None, diag
    real_vx /= norm
    real_vy /= norm

    diag["real_vx"] = round(float(real_vx), 6)
    diag["real_vy"] = round(float(real_vy), 6)

    # Dot product check ใน real-world space
    if len(track_points) >= 2:
        first_real = transform_point(track_points[0], homography_matrix)
        last_real  = transform_point(track_points[-1], homography_matrix)
        if first_real is not None and last_real is not None:
            dx = last_real[0] - first_real[0]
            dy = last_real[1] - first_real[1]
            displacement_norm = np.sqrt(dx**2 + dy**2)
            if displacement_norm > 0.1:
                dot_real = real_vx * dx + real_vy * dy
                diag["dot_product_real"] = round(float(dot_real), 4)
                if dot_real < 0:
                    real_vx, real_vy = -real_vx, -real_vy
                    diag["direction_flipped_real"] = True

    real_distance = speed_mps * n_seconds
    future_real_x = real_current[0] + real_vx * real_distance
    future_real_y = real_current[1] + real_vy * real_distance

    H_inv = np.linalg.inv(homography_matrix)
    future_pt = np.array([[[future_real_x, future_real_y]]], dtype=np.float32)
    future_pixel = cv2.perspectiveTransform(future_pt, H_inv)

    fp = (int(future_pixel[0][0][0]), int(future_pixel[0][0][1]))

    # [FIX] Sanity check: ถ้า future_point ไกลเกินไป → homography ระเบิด
    MAX_PIXEL_DISTANCE = 500
    dx_fp = fp[0] - point[0]
    dy_fp = fp[1] - point[1]
    pixel_dist = np.sqrt(dx_fp**2 + dy_fp**2)
    if pixel_dist > MAX_PIXEL_DISTANCE:
        diag["prediction_method"] = "homography_rejected"
        diag["future_point"] = fp  # เก็บค่าที่ระเบิดไว้ใน log เพื่อวิเคราะห์
        return None, diag

    diag["future_point"] = fp
    diag["prediction_method"] = "homography"
    return fp, diag


# ==================== User Input Helpers ====================

def select_classes(model):
    """แสดง class list จาก model แล้วให้ user เลือก"""
    class_names = model.names  # dict: {0: 'Bike', 1: 'Cars', ...}
    print("\n=== Available Classes (from model) ===")
    for class_id, class_name in class_names.items():
        print(f"  {class_id}: {class_name}")

    print("\nSelect classes to log (comma-separated IDs, or 'all' for all classes)")
    print(f"  Example: 0,1,3  or  all")

    while True:
        choice = input("Classes: ").strip()
        if choice.lower() == 'all':
            selected = set(class_names.keys())
            print(f"  ✓ Selected all classes: {list(class_names.values())}")
            return selected, class_names
        try:
            selected = set(int(x.strip()) for x in choice.split(','))
            valid = all(cid in class_names for cid in selected)
            if valid and len(selected) > 0:
                selected_names = [class_names[cid] for cid in selected]
                print(f"  ✓ Selected: {selected_names}")
                return selected, class_names
            else:
                print("  ✗ Invalid class ID(s). Please try again.")
        except ValueError:
            print("  ✗ Invalid format. Use comma-separated numbers or 'all'.")


def select_frame_range(total_frames, fps):
    """ให้ user เลือก frame range หรือช่วงเวลา"""
    total_sec = total_frames / fps
    total_min = int(total_sec // 60)
    total_sec_remainder = total_sec % 60

    print(f"\n=== Select Range ===")
    print(f"Video info: {total_frames} frames, {fps:.1f} FPS, duration: {total_min}:{total_sec_remainder:05.2f}")
    print(f"Choose input method:")
    print(f"  1. By frame range  (e.g. 100-500)")
    print(f"  2. By time range   (e.g. 0:05-0:30)")
    print(f"  3. All frames")

    while True:
        choice = input("\nChoice (1/2/3): ").strip()
        if choice == '1':
            while True:
                try:
                    range_str = input("Frame range (start-end): ").strip()
                    parts = range_str.split('-')
                    start_frame = int(parts[0].strip())
                    end_frame = int(parts[1].strip())
                    if 0 < start_frame <= end_frame <= total_frames:
                        print(f"  ✓ Frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
                        return start_frame, end_frame
                    else:
                        print(f"  ✗ Invalid range. Must be 1-{total_frames}")
                except (ValueError, IndexError):
                    print("  ✗ Invalid format. Use: start-end (e.g. 100-500)")

        elif choice == '2':
            while True:
                try:
                    range_str = input("Time range (M:SS-M:SS, e.g. 0:05-0:30): ").strip()
                    parts = range_str.split('-')
                    
                    def parse_time(t):
                        t = t.strip()
                        if ':' in t:
                            m, s = t.split(':')
                            return int(m) * 60 + float(s)
                        return float(t)
                    
                    start_sec = parse_time(parts[0])
                    end_sec = parse_time(parts[1])
                    start_frame = max(1, int(start_sec * fps))
                    end_frame = min(total_frames, int(end_sec * fps))

                    if start_frame <= end_frame:
                        print(f"  ✓ Time {start_sec:.1f}s to {end_sec:.1f}s → Frames {start_frame}-{end_frame}")
                        return start_frame, end_frame
                    else:
                        print("  ✗ Invalid time range.")
                except (ValueError, IndexError):
                    print("  ✗ Invalid format. Use: M:SS-M:SS (e.g. 0:05-0:30)")

        elif choice == '3':
            print(f"  ✓ All frames: 1 to {total_frames}")
            return 1, total_frames
        else:
            print("  ✗ Invalid choice.")


# ==================== Main ====================

if __name__ == "__main__":
    print("=" * 60)
    print("  Debug Logger — Collision Risk Detector v0.1.2")
    print("  เก็บ log ละเอียดสำหรับวิเคราะห์ bug เส้นทำนายทิศทาง")
    print("=" * 60)

    # 1. เลือกวิดีโอ
    video_path = select_video_file()
    cap = cv2.VideoCapture(video_path)
    model = YOLO(r"models/best.pt")

    REAL_FPS = get_fps.get_video_fps_cv2(video_path=video_path)
    PROCESS_EVERY_N_FRAME = 1
    FPS = REAL_FPS / PROCESS_EVERY_N_FRAME
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path}")
    print(f"FPS: {REAL_FPS}, Total frames: {TOTAL_FRAMES}")

    # 2. เลือก class จาก model
    selected_class_ids, class_names_dict = select_classes(model)

    # 3. เลือก frame range
    START_FRAME, END_FRAME = select_frame_range(TOTAL_FRAMES, REAL_FPS)

    # 4. Calibration
    try:
        calibrator = CalibrationTool()
        HOMOGRAPHY_MATRIX, CALIBRATION_POLYGON = calibrator.run(video_path=video_path)
    except Exception as e:
        print(f"Calibration error: {e}")
        HOMOGRAPHY_MATRIX = None
        CALIBRATION_POLYGON = None

    print(f"\nHomography matrix: {HOMOGRAPHY_MATRIX is not None}")
    print(f"Calibration polygon: {CALIBRATION_POLYGON is not None}")

    if not cap.isOpened():
        print("Error: could not open video file.")
        exit()

    # 5. Setup tracking
    byte_track = sv.ByteTrack(frame_rate=int(FPS))
    trace_annotator = sv.TraceAnnotator(trace_length=int(FPS*2), thickness=2, position=sv.Position.BOTTOM_CENTER)

    RUNID = get_runid()
    OUTPUTDIR = create_outputfolder(RUNID)

    speed_memory = defaultdict(lambda: deque(maxlen=5))
    direction_memory = defaultdict(lambda: deque(maxlen=20))
    object_in_zone = defaultdict(bool)
    last_speed = defaultdict(float)
    speed_history = defaultdict(list)

    # 6. Logging data
    log_records = []
    frame_count = 0

    print(f"\n{'='*60}")
    print(f"  Starting debug run: frames {START_FRAME}-{END_FRAME}")
    print(f"  Monitoring classes: {[class_names_dict[cid] for cid in selected_class_ids]}")
    print(f"{'='*60}\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # ข้ามถ้ายังไม่ถึง start frame (แต่ยังต้องให้ ByteTrack เห็น detection)
        if frame_count < START_FRAME:
            continue

        # หยุดเมื่อเกิน end frame
        if frame_count > END_FRAME:
            break

        if frame_count % PROCESS_EVERY_N_FRAME != 0:
            continue

        time_sec = frame_count / REAL_FPS

        # วาดพื้นที่ calibration
        if CALIBRATION_POLYGON is not None:
            overlay = frame.copy()
            cv2.polylines(overlay, [CALIBRATION_POLYGON], True, (0, 255, 255), 2)
            cv2.fillPoly(overlay, [CALIBRATION_POLYGON], (0, 255, 255))
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

        results = model.predict(source=frame, stream=True, conf=0.65)
        result = next(results)

        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(int)
        annotated_frame = trace_annotator.annotate(scene=result.plot(), detections=detections)

        if CALIBRATION_POLYGON is not None:
            cv2.polylines(annotated_frame, [CALIBRATION_POLYGON], True, (0, 255, 255), 2)

        for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
            if tracker_id is None:
                continue

            # กรอง class
            class_id = detections.class_id[i]
            if class_id not in selected_class_ids:
                continue

            class_name = class_names_dict.get(class_id, "unknown")
            in_zone = is_point_in_polygon(point, CALIBRATION_POLYGON)
            near_edge = is_near_frame_edge(point, annotated_frame.shape)

            # เก็บ track points
            direction_memory[tracker_id].append(point)

            # คำนวณทิศทาง (with diagnostics)
            angle, dir_vec, dir_diag = estimate_direction_debug(direction_memory[tracker_id])

            # คำนวณ projection (with diagnostics)
            proj_diag = {
                "real_vx": None, "real_vy": None,
                "dot_product_real": None, "direction_flipped_real": False,
                "future_point": None, "prediction_method": "none",
            }

            if dir_vec is not None and not near_edge:
                current_speed_mps = last_speed[tracker_id] / 3.6
                # [FIX] ใช้ homography เฉพาะเมื่ออยู่ใน zone
                use_homography = HOMOGRAPHY_MATRIX if in_zone else None
                if current_speed_mps > 0 and use_homography is not None:
                    future_point, proj_diag = project_future_point_debug(
                        point, dir_vec, current_speed_mps, 1,
                        use_homography, list(direction_memory[tracker_id])
                    )
                    if future_point is not None:
                        cv2.line(annotated_frame, (int(point[0]), int(point[1])),
                                future_point, (0, 0, 255), 2)
                else:
                    # Fallback
                    vx, vy, x0, y0 = dir_vec
                    end_x = int(point[0] + vx * 60)
                    end_y = int(point[1] + vy * 60)
                    cv2.line(annotated_frame, (int(point[0]), int(point[1])),
                            (end_x, end_y), (0, 0, 255), 2)
                    proj_diag["prediction_method"] = "fallback"
                    proj_diag["future_point"] = (end_x, end_y)

            # Speed calculation (เหมือน v0.1.2)
            if in_zone:
                speed_memory[tracker_id].append(point)
                object_in_zone[tracker_id] = True
                if len(speed_memory[tracker_id]) >= 2 and HOMOGRAPHY_MATRIX is not None:
                    p1 = speed_memory[tracker_id][0]
                    p2 = speed_memory[tracker_id][-1]
                    real_distance = calculate_real_distance(p1, p2, HOMOGRAPHY_MATRIX)
                    if real_distance is not None and real_distance > 0:
                        num_frames = len(speed_memory[tracker_id]) - 1
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
            else:
                if object_in_zone[tracker_id]:
                    if len(speed_history[tracker_id]) > 0:
                        speeds = np.array(speed_history[tracker_id])
                        q1, q3 = np.percentile(speeds, [25, 75])
                        iqr = q3 - q1
                        filtered = speeds[(speeds >= q1 - 1.5*iqr) & (speeds <= q3 + 1.5*iqr)]
                        if len(filtered) > 0:
                            last_speed[tracker_id] = np.median(filtered)
                    speed_memory[tracker_id].clear()
                    speed_history[tracker_id].clear()
                    object_in_zone[tracker_id] = False

            # สร้าง log record
            track_pts_list = [[int(p[0]), int(p[1])] for p in direction_memory[tracker_id]]
            speed_mem_list = [[int(p[0]), int(p[1])] for p in speed_memory[tracker_id]]

            record = {
                "frame_num": frame_count,
                "time_sec": round(time_sec, 3),
                "tracker_id": int(tracker_id),
                "class_id": int(class_id),
                "class_name": class_name,
                "point_x": int(point[0]),
                "point_y": int(point[1]),
                "in_zone": bool(in_zone),
                "near_edge": bool(near_edge),
                "speed_kmh": round(float(last_speed[tracker_id]), 2),
                "num_track_points": len(track_pts_list),
                # Direction diagnostics (image space)
                "vx_raw": dir_diag["vx_raw"],
                "vy_raw": dir_diag["vy_raw"],
                "overall_dx": dir_diag["overall_dx"],
                "overall_dy": dir_diag["overall_dy"],
                "dot_product_image": dir_diag["dot_product_image"],
                "direction_flipped_image": dir_diag["direction_flipped_image"],
                # Direction diagnostics (real-world space)
                "real_vx": proj_diag["real_vx"],
                "real_vy": proj_diag["real_vy"],
                "dot_product_real": proj_diag["dot_product_real"],
                "direction_flipped_real": proj_diag["direction_flipped_real"],
                # Prediction
                "prediction_method": proj_diag["prediction_method"],
                "future_point_x": proj_diag["future_point"][0] if proj_diag["future_point"] else None,
                "future_point_y": proj_diag["future_point"][1] if proj_diag["future_point"] else None,
                # Track data (JSON only)
                "track_points": track_pts_list,
                "speed_memory_points": speed_mem_list,
            }
            log_records.append(record)

        # Frame info overlay
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        info_text = f"Frame: {frame_count}  Time: {minutes:02d}:{seconds:05.2f}"
        progress = f"  [{frame_count - START_FRAME + 1}/{END_FRAME - START_FRAME + 1}]"
        (tw, th), _ = cv2.getTextSize(info_text + progress, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)
        cv2.putText(annotated_frame, info_text + progress, (12, 12 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Debug Logger", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⚠ Stopped early by user (pressed 'q')")
            break

    cap.release()
    cv2.destroyAllWindows()

    # ==================== Export Logs ====================
    if len(log_records) == 0:
        print("\n⚠ No records to save.")
    else:
        # CSV (without track_points arrays — too large for CSV)
        csv_path = os.path.join(OUTPUTDIR, "debug_log.csv")
        csv_fields = [k for k in log_records[0].keys() if k not in ("track_points", "speed_memory_points")]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(log_records)

        # JSON (full data including track_points)
        json_path = os.path.join(OUTPUTDIR, "debug_log.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "video_path": video_path,
                    "fps": REAL_FPS,
                    "frame_range": [START_FRAME, END_FRAME],
                    "selected_classes": {int(k): class_names_dict[k] for k in selected_class_ids},
                    "homography_available": HOMOGRAPHY_MATRIX is not None,
                    "total_records": len(log_records),
                },
                "records": log_records,
            }, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"  ✅ Debug logging complete!")
        print(f"{'='*60}")
        print(f"  Total records: {len(log_records)}")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        print(f"\n  CSV ใช้ดูภาพรวม (เปิดใน Excel/pandas)")
        print(f"  JSON มี track_points + speed_memory ละเอียด")
        print(f"\n  🔍 วิธีหา bug:")
        print(f"     ดูคอลัมน์ 'direction_flipped_image' และ 'direction_flipped_real'")
        print(f"     ถ้าเป็น True = โปรแกรมตรวจพบว่าเส้นชี้ผิดทิศแล้วแก้")
        print(f"     ถ้าเป็น False แต่เส้นยังชี้ผิด = มีสาเหตุอื่น")
