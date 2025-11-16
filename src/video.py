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
    
    def calculate_scale(self, distances, real_distances):
        """คำนวณมาตราส่วน pixel to meter"""
        if len(distances) != 4 or len(real_distances) != 4:
            print("The actual distance on all 4 sides must be specified.")
            return None
        
        scales = []
        labels = ["top", "right", "bottom", "left"]
        
        print("\n=== scale calculation ===")
        for i in range(4):
            if real_distances[i] > 0:
                scale = real_distances[i] / distances[i]
                scales.append(scale)
                print(f"{labels[i]}: {distances[i]:.2f} px = {real_distances[i]:.2f} m → scale = {scale:.5f} m/px")
        
        avg_scale = np.mean(scales)
        print(f"\n>>> average scale: {avg_scale:.5f} meter/pixel <<<")
        print(f">>> using PIXEL_TO_METER = {avg_scale:.5f} <<<\n")
        
        return avg_scale
    
    def run(self, video_path):
        """เรียกใช้งาน calibration tool และคืนค่า PIXEL_TO_METER"""
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
        
        pixel_to_meter = None
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset
                self.points = []
                self.frame = self.original_frame.copy()
                cv2.imshow("Calibration", self.frame)
                print("\nReseted - Sart selecting a new point")
            elif key == ord('c') and len(self.points) == 4:
                # คำนวณระยะทาง
                # print("\n=== ระยะทางในหน่วยพิกเซล ===")
                print("\n=== distance as pixels ===")
                distances = self.calculate_distances()
                
                if distances:
                    print("\nPlease enter the actual distance (in meters) of each side:")
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
                    
                    # คำนวณมาตราส่วน
                    pixel_to_meter = self.calculate_scale(distances, real_distances)
                    
                    if pixel_to_meter:
                        print(f"\n✓ Success! PIXEL_TO_METER = {pixel_to_meter:.5f}")
                        print("Press 'q' to complete, or 'r' to reset\n")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return pixel_to_meter

def find_nearest_object(current_idx, points):
    current_point = points[current_idx]
    min_distance = float("inf")
    for idx, point in enumerate(points):
        if idx == current_idx:
            continue
        dx = point[0] - current_point[0]
        dy = point[1] - current_point[1]
        if dy > 0 and abs(dx) < 50:  # เฉพาะคันที่อยู่ด้านหน้าในแนวเดียวกัน
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

    byte_track = sv.ByteTrack(frame_rate=FPS)  # ปรับ fps ตามจริงของวิดีโอ
    trace_annotator = sv.TraceAnnotator(trace_length=FPS*2, thickness=2, position=sv.Position.BOTTOM_CENTER)

    RUNID = get_runid()
    print(RUNID)
    OUTPUTDIR = create_outputfolder(RUNID)

    try:
        calibrator = CalibrationTool()
        PIXEL_TO_METER = calibrator.run(video_path=video_path)
    except Exception as e:
        print(f"error: {e}")
    print(f"pixel to mater: {PIXEL_TO_METER}")


    if not cap.isOpened():
        print("Error: could not open video file.")
        exit()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.predict(source=frame, stream=True, conf = 0.65)

            # for result in results:
            result = next(results)

            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update_with_detections(detections=detections)

            # วาดเส้นทาง
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(int)
            annotated_frame = trace_annotator.annotate(scene=result.plot(), detections=detections)

            for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
                if tracker_id is None:
                    continue
                speed_memory[tracker_id].append(point)

                if len(speed_memory[tracker_id]) >= 2:

                    p1 = speed_memory[tracker_id][0]
                    p2 = speed_memory[tracker_id][-1]
                    
                    # dy = abs(speed_memory[tracker_id][-1] - speed_memory[tracker_id][0])
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    pixel_dist = np.hypot(dx, dy)
                    num_frames = len(speed_memory[tracker_id]) - 1
                    time_elapsed_s = num_frames / FPS
                    if time_elapsed_s > 0:
                        # speed = dy * 30 / len(speed_memory[tracker_id])
                        # speed_mps = speed * PIXEL_TO_METER

                        speed_mps = (pixel_dist / time_elapsed_s) * PIXEL_TO_METER
                        speed_kmh = speed_mps * 3.6

                        cv2.putText(annotated_frame, f"Speed: {speed_kmh:.1f} km/h",
                                    (point[0] - 50, point[1] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # nearest = find_nearest_object(i, points)
                    # if nearest is not None and speed > 0:
                    #     ttc = nearest / speed
                    #     if ttc < 2:
                    #         cv2.putText(annotated_frame, f"RISK #{tracker_id}", (point[0], point[1]-20),
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    #         handle_risk_event(
                    #             annotated_frame=annotated_frame,
                    #             output_dir=OUTPUTDIR,
                    #             detections=detections,
                    #             result=result,
                    #             tracker_id=tracker_id,
                    #             points=points,
                    #             i=i,
                    #             cap=cap,
                    #             save_log=False,
                    #             save_frame=False,
                    #         )
            cv2.imshow("YOLOv8 Object Detection", annotated_frame)

            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
        
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Video processing finished.")