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
                    if not line or line.startswith('#'):  # ข้ามบรรทัดว่างและคอมเมนต์
                        continue
                    
                    # รองรับทั้ง comma และ space
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
                    
                    # รองรับทั้ง comma และ space
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
        
        # วาดจุดและเลข
        for idx, pt in enumerate(self.points):
            cv2.circle(self.frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(self.frame, str(idx+1), (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # วาดเส้นเชื่อม
        if len(self.points) > 1:
            for i in range(len(self.points)-1):
                cv2.line(self.frame, self.points[i], self.points[i+1], (255, 0, 0), 2)
        
        # วาดเส้นปิดรูป
        if len(self.points) == 4:
            cv2.line(self.frame, self.points[3], self.points[0], (255, 0, 0), 2)
            self.polygon = np.array(self.points, dtype=np.int32)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            
            # วาดจุดและเส้น
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
        
        # จุดในภาพ (image coordinates)
        src_points = np.float32(self.points)
        
        # จุดในพื้นที่จริง (real-world coordinates in meters)
        # สร้างสี่เหลี่ยมจากระยะจริง โดยเริ่มที่ (0,0)
        width = real_distances[0]   # top (1-2)
        height = real_distances[1]  # right (2-3)
        
        # กำหนดจุดในโลกจริงเป็นสี่เหลี่ยมผืนผ้า
        dst_points = np.float32([
            [0, 0],           # top-left
            [width, 0],       # top-right
            [width, height],  # bottom-right
            [0, height]       # bottom-left
        ])
        
        # คำนวณ homography matrix
        H, status = cv2.findHomography(src_points, dst_points)
        
        print("\n=== Homography Matrix ===")
        print(H)
        print(f"Real-world area: {width:.2f}m x {height:.2f}m")
        
        return H
    
    def get_polygon(self):
        """คืนค่า polygon ของพื้นที่ที่เลือก"""
        if len(self.points) == 4:
            return np.array(self.points, dtype=np.int32)
        return None
    
    def get_homography(self):
        """คืนค่า homography matrix"""
        return self.homography_matrix
    
    def calibrate_a_zone(self, video_path, zone_name):
        """calibrate one zone - return {name, points, polygon, homography_matrix}"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: failed to open video")
            return None, None
        
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read a first frame")
            cap.release()
            return None, None

        self.original_frame = frame.copy()
        self.frame = frame.copy()
        # ให้ผู้ใช้เลือกวิธีการกำหนดจุด
        print("\n=== Calibration Tool ===")
        print("Choose input method:")
        print("  1. Draw with mouse (interactive)")
        print("  2. Load from txt file")
        print("  3. Type coordinates via keyboard")
        
        while True:
            try:
                choice = input("\nEnter your choice (1/2/3): ").strip()
                
                if choice == '1':
                    # วิธีเดิม - วาดด้วยเมาส์
                    print("\n=== Mouse Drawing Mode ===")
                    print("Instructions:")
                    print("1. Click 4 points to create a rectangle on the surface with known distances")
                    print("   (e.g., lane width, line length)")
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
                            # Reset
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
                    # โหลดจาก txt file
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
                    # พิมพ์พิกัดผ่าน keyboard
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
        
        # ตรวจสอบว่ามีจุดครบ 4 จุดหรือไม่
        if len(self.points) != 4:
            print("Error: Need exactly 4 points for calibration")
            cap.release()
            return None, None
        
        # คำนวณระยะทางและสร้าง homography
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

            # สร้าง homography matrix
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

def transform_point(point, homography_matrix):
    """แปลงจุดจากพิกัดภาพเป็นพิกัดจริง (เมตร)"""
    if homography_matrix is None:
        return None
    
    # แปลงจุดเป็น homogeneous coordinates
    pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    
    # ใช้ homography แปลงพิกัด
    transformed = cv2.perspectiveTransform(pt, homography_matrix)
    
    return transformed[0][0]

def calculate_real_distance(p1, p2, homography_matrix):
    """คำนวณระยะทางจริงระหว่าง 2 จุด (เมตร)"""
    if homography_matrix is None:
        return None
    
    # แปลงทั้ง 2 จุดเป็นพิกัดจริง
    real_p1 = transform_point(p1, homography_matrix)
    real_p2 = transform_point(p2, homography_matrix)
    
    if real_p1 is None or real_p2 is None:
        return None
    
    # คำนวณระยะห่าง (Euclidean distance)
    distance = np.sqrt((real_p2[0] - real_p1[0])**2 + (real_p2[1] - real_p1[1])**2)
    
    return distance

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

# ==================== v0.1.2.fix — Prediction Line Features ====================

def estimate_direction(track_points):
    """คำนวณทิศทางการเคลื่อนที่จาก track points ด้วย weighted average displacement
    
    FIX: เพิ่ม dot product check เทียบกับ overall displacement (จุดแรก→จุดสุดท้าย)
    เพื่อป้องกันไม่ให้ direction vector ชี้สวนทางกับการเคลื่อนที่จริง
    """
    if len(track_points) < 5:
        return None, None
    
    pts = list(track_points)
    n = len(pts)
    
    # คำนวณ displacement ระหว่างจุดติดกัน และให้ weight มากกับจุดใหม่
    total_vx, total_vy, total_w = 0.0, 0.0, 0.0
    for i in range(1, n):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        w = i  # จุดใหม่กว่าได้ weight มากกว่า
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
    
    # [FIX] ตรวจสอบทิศทางด้วย dot product เทียบกับ overall displacement
    # ใช้ displacement จากจุดแรก → จุดสุดท้ายเป็น ground truth ของทิศทาง
    overall_dx = float(pts[-1][0] - pts[0][0])
    overall_dy = float(pts[-1][1] - pts[0][1])
    overall_norm = np.sqrt(overall_dx**2 + overall_dy**2)
    
    if overall_norm > 1e-6:
        dot_product = vx * overall_dx + vy * overall_dy
        # ถ้า dot product ติดลบ แสดงว่า direction vector ชี้สวนทาง → กลับทิศ
        if dot_product < 0:
            vx = -vx
            vy = -vy
    
    # x0, y0 ใช้จุดล่าสุด
    x0, y0 = float(pts[-1][0]), float(pts[-1][1])
    
    angle = np.degrees(np.arctan2(vy, vx))
    return float(angle), (vx, vy, x0, y0)

def project_future_point(point, direction_vector, speed_mps, n_seconds, homography_matrix, track_points):
    """คาดคะเนตำแหน่งในอนาคตของวัตถุ
    
    FIX: เปิด dot product check ใน real-world space เพื่อป้องกัน
    homography transformation กลับทิศ direction vector
    """
    if homography_matrix is None or direction_vector is None or speed_mps <= 0:
        return None

    vx, vy, x0, y0 = direction_vector

    real_current = transform_point(point, homography_matrix)
    if real_current is None:
        return None

    # แปลง direction vector → real-world space
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

    # [FIX] ตรวจสอบทิศใน real-world space อีกครั้ง
    # โดยเปรียบเทียบกับ displacement จริงของ track_points
    if len(track_points) >= 2:
        first_real = transform_point(track_points[0], homography_matrix)
        last_real  = transform_point(track_points[-1], homography_matrix)
        if first_real is not None and last_real is not None:
            dx = last_real[0] - first_real[0]
            dy = last_real[1] - first_real[1]
            displacement_norm = np.sqrt(dx**2 + dy**2)
            # ตรวจสอบเฉพาะเมื่อมี displacement พอสมควร (ไม่ใช่แค่ jitter)
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

    # [FIX] Sanity check: ถ้า future_point อยู่ไกลจาก current point เกินไป
    # แสดงว่า homography projection ระเบิด → return None เพื่อใช้ fallback แทน
    MAX_PIXEL_DISTANCE = 500
    dx = result[0] - point[0]
    dy = result[1] - point[1]
    pixel_dist = np.sqrt(dx**2 + dy**2)
    if pixel_dist > MAX_PIXEL_DISTANCE:
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

    # Fallback: ใช้ direction vector ตรงๆ ใน pixel space
    vx, vy, x0, y0 = direction_vector
    end_x = int(point[0] + vx * fallback_length)
    end_y = int(point[1] + vy * fallback_length)
    cv2.line(frame, (int(point[0]), int(point[1])), (end_x, end_y), color, thickness)

def is_near_frame_edge(point, frame_shape, margin=50):
    """ถ้า object อยู่ใกล้ขอบ frame ภายใน margin pixels ให้ถือว่ากำลังออก"""
    h, w = frame_shape[:2]
    x, y = point
    return x < margin or x > w - margin or y < margin or y > h - margin

# ==================== Main ====================

if __name__ == "__main__":
    video_path = select_video_file()
    cap = cv2.VideoCapture(video_path)
    speed_memory = defaultdict(lambda: deque(maxlen=5))
    direction_memory = defaultdict(lambda: deque(maxlen=20))
    model = YOLO(r"models/best.pt")

    REAL_FPS = get_fps.get_video_fps_cv2(video_path=video_path)
    PROCESS_EVERY_N_FRAME = 1
    print(f"fps: {REAL_FPS}")
    FPS = REAL_FPS/PROCESS_EVERY_N_FRAME
    frame_count = 0

    byte_track = sv.ByteTrack(frame_rate=int(FPS))
    trace_annotator = sv.TraceAnnotator(trace_length=int(FPS*2), thickness=2, position=sv.Position.BOTTOM_CENTER)

    RUNID = get_runid()
    print(RUNID)
    # OUTPUTDIR = create_outputfolder(RUNID)

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

    # ตัวแปรเก็บสถานะว่าวัตถุอยู่ในพื้นที่หรือไม่
    object_in_zone = defaultdict(bool)
    # ตัวแปรเก็บความเร็วล่าสุดของแต่ละวัตถุ
    last_speed = defaultdict(float)
    # ตัวแปรเก็บประวัติความเร็วทั้งหมดในโซน (สำหรับหาค่าเฉลี่ย/median)
    speed_history = defaultdict(list)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count = frame_count + 1
        if frame_count % PROCESS_EVERY_N_FRAME != 0:
            continue

        # วาดพื้นที่ calibration บนเฟรม (โปร่งแสง)
        # if CALIBRATION_POLYGON is not None:
        #     overlay = frame.copy()
        #     cv2.polylines(overlay, [CALIBRATION_POLYGON], True, (0, 255, 255), 2)
        #     cv2.fillPoly(overlay, [CALIBRATION_POLYGON], (0, 255, 255))
        #     frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

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

        # วาดเส้นทาง
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(int)
        annotated_frame = trace_annotator.annotate(scene=result.plot(), detections=detections)

        # วาดพื้นที่ calibration บน annotated_frame
        for zone_idx, zone in enumerate(ZONES):
            color = (0, 255, 255)
            cv2.polylines(annotated_frame, [zone['polygon']], True, color, 2)

        for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
            if tracker_id is None:
                continue
            
            # ตรวจสอบว่าจุดอยู่ในพื้นที่ calibration หรือไม่
            # in_zone = is_point_in_polygon(point, CALIBRATION_POLYGON)
            matched_zones = []
            for zone_idx, zone in enumerate(ZONES):
                if is_point_in_polygon(point, zone['polygon']):
                    matched_zones.append((zone_idx, zone))

            in_zone = len(matched_zones) > 0
            
            # เก็บ track points สำหรับคำนวณทิศทาง
            direction_memory[tracker_id].append(point)
            angle, dir_vec = estimate_direction(direction_memory[tracker_id])
            if dir_vec is not None and not is_near_frame_edge(point, annotated_frame.shape):
                current_speed_mps = last_speed[tracker_id] / 3.6
                # [FIX] ใช้ homography เฉพาะเมื่ออยู่ใน calibration zone
                # นอก zone → homography projection จะระเบิด → ใช้ fallback แทน

                # use_homography = HOMOGRAPHY_MATRIX if in_zone else None
                use_homography = matched_zones[0][1]['homography_matrix'] if matched_zones else None
                draw_direction_line(
                    annotated_frame, point, dir_vec,
                    track_points=list(direction_memory[tracker_id]),
                    speed_mps=current_speed_mps,
                    n_seconds=1,
                    homography_matrix=use_homography,
                    color=(0,0,255), thickness=2
                )



            # อัพเดท speed_memory เฉพาะเมื่ออยู่ในโซน
            if in_zone:
                for zone_idx, zone in matched_zones:
                    key = (tracker_id, zone_idx)
                    speed_memory[key].append(point)
                    object_in_zone[key] = True
                    
                    # if len(speed_memory[tracker_id]) >= 2 and HOMOGRAPHY_MATRIX is not None:
                    #     p1 = speed_memory[tracker_id][0]
                    #     p2 = speed_memory[tracker_id][-1]
                    if len(speed_memory[key]) >= 2:
                        p1 = speed_memory[key][0]
                        p2 = speed_memory[key][-1]
                        
                        # ใช้ homography คำนวณระยะทางจริง
                        real_distance = calculate_real_distance(p1, p2, zone['homography_matrix'])
                        
                        if real_distance is not None and real_distance > 0:
                            num_frames = len(speed_memory[key]) - 1
                            time_elapsed_s = num_frames / FPS
                            
                            if time_elapsed_s > 0:
                                speed_mps = real_distance / time_elapsed_s
                                speed_kmh = speed_mps * 3.6
                                
                                # เก็บประวัติความเร็วทั้งหมด
                                speed_history[tracker_id].append(speed_kmh)
                                
                                # ใช้ค่า median จากประวัติความเร็ว (ทนทานต่อค่าผิดปกติ)
                                if len(speed_history[tracker_id]) >= 3:
                                    display_speed = np.median(speed_history[tracker_id])
                                else:
                                    display_speed = speed_kmh
                                
                                # อัพเดทความเร็วล่าสุด
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
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            filtered_speeds = speeds[(speeds >= lower_bound) & (speeds <= upper_bound)]
                            if len(filtered_speeds) > 0:
                                final_speed = np.median(filtered_speeds)
                                last_speed[tracker_id] = final_speed

                        speed_memory[key].clear()
                        speed_history[tracker_id].clear()
                        object_in_zone[key] = False

                # แสดงความเร็วล่าสุด (ถ้ามี)
                if tracker_id in last_speed and last_speed[tracker_id] > 0:
                    cv2.putText(annotated_frame, f"Speed: {last_speed[tracker_id]:.1f} km/h",
                                (point[0] - 50, point[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        # แสดง frame info overlay (ถ้าเปิด debug)
        if SHOW_FRAME_INFO:
            time_sec = frame_count / REAL_FPS
            minutes = int(time_sec // 60)
            seconds = time_sec % 60
            info_text = f"Frame: {frame_count}  Time: {minutes:02d}:{seconds:05.2f}"
            # วาดพื้นหลังสีดำโปร่งแสง
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
