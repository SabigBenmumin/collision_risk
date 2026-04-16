import cv2
from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
from collections import defaultdict, deque
from filemanage import select_video_file, get_runid, create_outputfolder, handle_risk_event

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class PerspectiveCalibrationTool:
    def __init__(self):
        self.points = []
        self.frame = None
        self.original_frame = None
        self.homography_matrix = None
        self.inverse_homography = None
        
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
            
            cv2.imshow("Calibration", self.frame)
    
    def calculate_homography(self, real_width, real_height):
        """คำนวณ homography matrix สำหรับ perspective transform"""
        if len(self.points) != 4:
            print("ต้องมีจุด 4 จุดเท่านั้น!")
            return None
        
        # จุดต้นทาง (จากภาพ) - ตามลำดับที่คลิก
        src_points = np.float32(self.points)
        
        # จุดปลายทาง (ในพื้นที่จริง bird's eye view)
        # สร้างสี่เหลี่ยมผืนผ้าที่มีสัดส่วนตามขนาดจริง
        dst_points = np.float32([
            [0, 0],                      # top-left
            [real_width, 0],             # top-right
            [real_width, real_height],   # bottom-right
            [0, real_height]             # bottom-left
        ])
        
        # คำนวณ homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_homography = cv2.getPerspectiveTransform(dst_points, src_points)
        
        print(f"\n✓ Homography matrix คำนวณเสร็จสิ้น")
        print(f"พื้นที่จริง: {real_width}m x {real_height}m")
        
        return self.homography_matrix
    
    def transform_point(self, point):
        """แปลงจุดจากภาพเดิมไปเป็นพิกัดใน bird's eye view (หน่วยเมตร)"""
        if self.homography_matrix is None:
            return None
        
        # แปลง point เป็น homogeneous coordinates
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        
        return transformed[0][0]
    
    def visualize_birdseye(self, frame, detections, points):
        """แสดงภาพ bird's eye view"""
        if self.homography_matrix is None:
            return None
        
        # สร้างภาพ bird's eye view
        h, w = frame.shape[:2]
        birdseye = cv2.warpPerspective(frame, self.homography_matrix, (400, 600))
        
        # วาดจุดของวัตถุใน bird's eye view
        for point in points:
            transformed_pt = self.transform_point(point)
            if transformed_pt is not None:
                x, y = int(transformed_pt[0]), int(transformed_pt[1])
                if 0 <= x < 400 and 0 <= y < 600:
                    cv2.circle(birdseye, (x, y), 5, (0, 255, 255), -1)
        
        # เพิ่ม grid เพื่อแสดงระยะ
        for i in range(0, 600, 100):  # ทุกๆ 100 pixels (ปรับตามขนาดจริง)
            cv2.line(birdseye, (0, i), (400, i), (100, 100, 100), 1)
        for i in range(0, 400, 100):
            cv2.line(birdseye, (i, 0), (i, 600), (100, 100, 100), 1)
        
        return birdseye
    
    def run(self, video_path):
        """เรียกใช้งาน calibration tool"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: ไม่สามารถเปิดวิดีโอได้")
            return None
        
        ret, frame = cap.read()
        if not ret:
            print("Error: ไม่สามารถอ่านเฟรมแรกได้")
            cap.release()
            return None
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        
        print("\n=== Perspective Calibration Tool ===")
        print("คำแนะนำ:")
        print("1. คลิก 4 จุดเพื่อสร้างสี่เหลี่ยมบนพื้นผิวที่รู้ระยะทางจริง")
        print("   (เช่น ความกว้างเลน หรือพื้นที่ที่รู้ขนาด)")
        print("2. คลิกตามลำดับ: บนซ้าย -> บนขวา -> ล่างขวา -> ล่างซ้าย")
        print("3. กด 'c' เมื่อเลือกครบ 4 จุด")
        print("4. กด 'r' เพื่อเริ่มใหม่")
        print("5. กด 'q' เพื่อเสร็จสิ้นการ calibrate\n")
        
        cv2.imshow("Calibration", self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset
                self.points = []
                self.frame = self.original_frame.copy()
                cv2.imshow("Calibration", self.frame)
                print("\nรีเซ็ตแล้ว - เริ่มเลือกจุดใหม่")
            elif key == ord('c') and len(self.points) == 4:
                print("\n=== กำหนดขนาดจริงของพื้นที่ ===")
                
                try:
                    real_width = float(input("ความกว้างจริง (เมตร) ระหว่างจุด 1-2 และ 4-3: "))
                    real_height = float(input("ความยาวจริง (เมตร) ระหว่างจุด 1-4 และ 2-3: "))
                    
                    self.calculate_homography(real_width, real_height)
                    
                    if self.homography_matrix is not None:
                        print(f"\n✓ สำเร็จ! Perspective calibration เสร็จสิ้น")
                        print("กด 'q' เพื่อดำเนินการต่อ หรือ 'r' เพื่อเริ่มใหม่\n")
                        
                except ValueError:
                    print("กรุณาใส่ตัวเลข!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.homography_matrix

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
    transformed_position_memory = defaultdict(lambda: deque(maxlen=5))
    
    model = YOLO(r"models/best.pt")
    FPS = 25

    byte_track = sv.ByteTrack(frame_rate=FPS)
    trace_annotator = sv.TraceAnnotator(trace_length=FPS*2, thickness=2, position=sv.Position.BOTTOM_CENTER)

    RUNID = get_runid()
    print(RUNID)
    OUTPUTDIR = create_outputfolder(RUNID)

    # Perspective Calibration
    try:
        calibrator = PerspectiveCalibrationTool()
        homography_matrix = calibrator.run(video_path=video_path)
        
        if homography_matrix is None:
            print("Warning: ไม่ได้ทำ calibration จะใช้การคำนวณแบบเดิม")
    except Exception as e:
        print(f"Error during calibration: {e}")
        homography_matrix = None
        calibrator = None

    if not cap.isOpened():
        print("Error: could not open video file.")
        exit()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.predict(source=frame, stream=True, conf=0.65)
            result = next(results)

            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(int)
            annotated_frame = trace_annotator.annotate(scene=result.plot(), detections=detections)

            for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
                if tracker_id is None:
                    continue
                
                # ใช้ perspective transform ถ้ามี
                if homography_matrix is not None and calibrator is not None:
                    transformed_pt = calibrator.transform_point(point)
                    if transformed_pt is not None:
                        transformed_position_memory[tracker_id].append(transformed_pt)
                        
                        if len(transformed_position_memory[tracker_id]) >= 2:
                            p1 = transformed_position_memory[tracker_id][0]
                            p2 = transformed_position_memory[tracker_id][-1]
                            
                            # คำนวณระยะทางจริงใน bird's eye view (หน่วยเมตร)
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            distance_meters = np.hypot(dx, dy)
                            
                            num_frames = len(transformed_position_memory[tracker_id]) - 1
                            time_elapsed_s = num_frames / FPS
                            
                            if time_elapsed_s > 0:
                                speed_mps = distance_meters / time_elapsed_s
                                speed_kmh = speed_mps * 3.6
                                
                                cv2.putText(annotated_frame, f"Speed: {speed_kmh:.1f} km/h",
                                           (point[0] - 50, point[1] - 40),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # ถ้าไม่มี calibration ใช้วิธีเดิม
                    speed_memory[tracker_id].append(point)
                    
                    if len(speed_memory[tracker_id]) >= 2:
                        print("Warning: กำลังใช้การคำนวณแบบไม่มี perspective correction")

            # แสดง bird's eye view (ถ้ามี)
            if calibrator is not None and homography_matrix is not None:
                birdseye = calibrator.visualize_birdseye(frame, detections, points)
                if birdseye is not None:
                    cv2.imshow("Bird's Eye View", birdseye)

            cv2.imshow("YOLOv8 Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Video processing finished.")
