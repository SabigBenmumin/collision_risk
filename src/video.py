import cv2
from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
from collections import defaultdict, deque
from filemanage import select_video_file, get_runid, create_outputfolder, handle_risk_event

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

video_path = select_video_file()
cap = cv2.VideoCapture(video_path)
speed_memory = defaultdict(lambda: deque(maxlen=5))
model = YOLO(r"models/best.pt")

byte_track = sv.ByteTrack(frame_rate=30)  # ปรับ fps ตามจริงของวิดีโอ
trace_annotator = sv.TraceAnnotator(trace_length=60, thickness=2)

RUNID = get_runid()
print(RUNID)
OUTPUTDIR = create_outputfolder(RUNID)

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
            speed_memory[tracker_id].append(point[1])
            if len(speed_memory[tracker_id]) >= 2:
                dy = abs(speed_memory[tracker_id][-1] - speed_memory[tracker_id][0])
                speed = dy * 30 / len(speed_memory[tracker_id])
                nearest = find_nearest_object(i, points)
                if nearest is not None and speed > 0:
                    ttc = nearest / speed
                    if ttc < 5:
                        cv2.putText(annotated_frame, f"RISK #{tracker_id}", (point[0], point[1]-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        handle_risk_event(
                            annotated_frame=annotated_frame,
                            output_dir=OUTPUTDIR,
                            detections=detections,
                            result=result,
                            tracker_id=tracker_id,
                            points=points,
                            i=i,
                            cap=cap,
                            save_log=True,
                            save_frame=True,
                        )
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
       
        break

cap.release()
cv2.destroyAllWindows()

print("Video processing finished.")