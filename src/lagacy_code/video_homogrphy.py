import cv2
from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
from collections import defaultdict, deque
from filemanage import select_video_file, get_runid, create_outputfolder, handle_risk_event

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class CalibrationTool:
    def __init__(self):
        self.points = []
        self.frame = None
        self.original_frame = None
        self.polygon = None
        self.homography_matrix = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            
            # วาดจุดและเลขบนภาพ
            self.frame = self.original_frame.copy()
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
            
            cv2.imshow("Calibration", self.frame)
    
    def calculate_distances(self):
        """คำนวดระยะทางในหน่วยพิกเซลของแต่ละด้าน"""
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
    
    def run(self, video_path):
        """เรียกใช้งาน calibration tool และคืนค่า homography matrix และ polygon"""
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
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        
        print("\n=== Calibration tools ===")
        print("Instructions:")
        print("1. Click 4 points to create a rectangle on the surface with known distances")
        print("   (e.g., lane width, line length)")
        print("2. Click in the following order: top left -> top right -> bottom right -> bottom left")
        print("3. Press 'c' when all 4 points are selected")
        print("4. Press 'r' to restart")
        print("5. Press 'q' to complete the calibration\n")
        
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
                # คำนวดระยะทาง
                print("\n=== distance as pixels ===")
                distances = self.calculate_distances()
                
                if distances:
                    print("\nPlease enter the actual distance (in meters) of each side:")
                    print("NOTE: For homography, we mainly use top (width) and right (height)")
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
                    
                    # สร้าง homography matrix
                    self.homography_matrix = self.create_homography(real_distances)
                    
                    if self.homography_matrix is not None:
                        print(f"\n✓ Success! Homography matrix created")
                        print("This will accurately handle perspective distortion")
                        print("Press 'q' to complete, or 'r' to reset\n")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.get_homography(), self.get_polygon()

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

if __name__ == "__main__":
    video_path = select_video_file()
    cap = cv2.VideoCapture(video_path)
    speed_memory = defaultdict(lambda: deque(maxlen=5))
    model = YOLO(r"models/best.pt")
    FPS = 25

    byte_track = sv.ByteTrack(frame_rate=FPS)
    trace_annotator = sv.TraceAnnotator(trace_length=FPS*2, thickness=2, position=sv.Position.BOTTOM_CENTER)

    RUNID = get_runid()
    print(RUNID)
    OUTPUTDIR = create_outputfolder(RUNID)

    try:
        calibrator = CalibrationTool()
        HOMOGRAPHY_MATRIX, CALIBRATION_POLYGON = calibrator.run(video_path=video_path)
    except Exception as e:
        print(f"error: {e}")
        HOMOGRAPHY_MATRIX = None
        CALIBRATION_POLYGON = None
    
    print(f"Homography matrix: {HOMOGRAPHY_MATRIX is not None}")
    print(f"Calibration polygon: {CALIBRATION_POLYGON is not None}")

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
        if success:
            # วาดพื้นที่ calibration บนเฟรม (โปร่งแสง)
            if CALIBRATION_POLYGON is not None:
                overlay = frame.copy()
                cv2.polylines(overlay, [CALIBRATION_POLYGON], True, (0, 255, 255), 2)
                cv2.fillPoly(overlay, [CALIBRATION_POLYGON], (0, 255, 255))
                frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            
            results = model.predict(source=frame, stream=True, conf=0.65)
            result = next(results)

            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update_with_detections(detections=detections)

            # วาดเส้นทาง
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(int)
            annotated_frame = trace_annotator.annotate(scene=result.plot(), detections=detections)

            # วาดพื้นที่ calibration บน annotated_frame
            if CALIBRATION_POLYGON is not None:
                cv2.polylines(annotated_frame, [CALIBRATION_POLYGON], True, (0, 255, 255), 2)

            for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
                if tracker_id is None:
                    continue
                
                # ตรวจสอบว่าจุดอยู่ในพื้นที่ calibration หรือไม่
                in_zone = is_point_in_polygon(point, CALIBRATION_POLYGON)
                
                # อัพเดท speed_memory เฉพาะเมื่ออยู่ในโซน
                if in_zone:
                    speed_memory[tracker_id].append(point)
                    object_in_zone[tracker_id] = True
                    
                    if len(speed_memory[tracker_id]) >= 2 and HOMOGRAPHY_MATRIX is not None:
                        p1 = speed_memory[tracker_id][0]
                        p2 = speed_memory[tracker_id][-1]
                        
                        # ใช้ homography คำนวณระยะทางจริง
                        real_distance = calculate_real_distance(p1, p2, HOMOGRAPHY_MATRIX)
                        
                        if real_distance is not None and real_distance > 0:
                            num_frames = len(speed_memory[tracker_id]) - 1
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
                    # ถ้าเพิ่งออกจากโซน ให้คำนวณความเร็วเฉลี่ยจากประวัติทั้งหมด
                    if object_in_zone[tracker_id]:
                        # ใช้ค่า median ของความเร็วทั้งหมดที่วัดได้ในโซน
                        if len(speed_history[tracker_id]) > 0:
                            # กรองค่าผิดปกติออก (ใช้เฉพาะค่าที่อยู่ใน IQR)
                            speeds = np.array(speed_history[tracker_id])
                            q1 = np.percentile(speeds, 25)
                            q3 = np.percentile(speeds, 75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            # กรองค่าที่อยู่นอก IQR
                            filtered_speeds = speeds[(speeds >= lower_bound) & (speeds <= upper_bound)]
                            
                            if len(filtered_speeds) > 0:
                                # ใช้ค่า median ของความเร็วที่ผ่านการกรอง
                                final_speed = np.median(filtered_speeds)
                                last_speed[tracker_id] = final_speed
                        
                        # ล้าง memory
                        speed_memory[tracker_id].clear()
                        speed_history[tracker_id].clear()
                        object_in_zone[tracker_id] = False
                    
                    # แสดงความเร็วล่าสุด (ถ้ามี)
                    if tracker_id in last_speed and last_speed[tracker_id] > 0:
                        cv2.putText(annotated_frame, f"Speed: {last_speed[tracker_id]:.1f} km/h",
                                    (point[0] - 50, point[1] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)  # สีส้ม

            cv2.imshow("YOLOv8 Object Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Video processing finished.")