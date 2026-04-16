
"""
detection with GUI
new output file format

"""
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog
import ast
from PIL import Image, ImageTk

if getattr(sys, 'frozen', False):
    # ถ้ารันจาก .exe (build ด้วย PyInstaller)
    BASE_PATH = sys._MEIPASS
else:
    # ถ้ารันปกติจากโค้ด Python
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

detectModel = os.path.join(BASE_PATH, r"models/best.pt")
icon_path = os.path.join(BASE_PATH, r"icon/StopSense.ico")

def classify_zone(point, target_height):
    x, y = point
    return "A" if y < target_height // 2 else "B"


def find_nearest_object(current_idx, points, axis="y"):
    current_point = points[current_idx]
    min_distance = float("inf")
    for idx, point in enumerate(points):
        if idx == current_idx:
            continue
        dx = point[0] - current_point[0]
        dy = point[1] - current_point[1]
        if axis == "y" and dy > 0 and abs(dx) < 2:
            distance = np.hypot(dx, dy)
            if distance < min_distance:
                min_distance = distance
        elif axis == "x" and dx > 0 and abs(dy) < 2:
            distance = np.hypot(dx, dy)
            if distance < min_distance:
                min_distance = distance
    return min_distance if min_distance != float("inf") else None


class ViewsTranformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        if point is None or len(point) == 0:
            return None
        if point.ndim != 2 or point.shape[1] != 2:
            raise ValueError("Input point must be a 2D array with shape (N, 2)")
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_points = cv2.perspectiveTransform(reshaped_point, self.m)
        return transform_points.reshape(-1, 2)


def initialize_components(video_info, model_path):
    model = YOLO(model_path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    return {
        "model": model,
        "byte_track": byte_track,
        "bbox_annotator": sv.BoxAnnotator(
            thickness=thickness, color_lookup=sv.ColorLookup.TRACK
        ),
        "label_annotator": sv.LabelAnnotator(
            text_scale=text_scale,
            # text_thickness=1,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK,
        ),
        "trace_annotator": sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK,
        ),
    }


def annotate_and_write(
    frame,
    detections,
    labels,
    components,
    source_polygon,
    preview_ratio,
    show_preview,
    update_preview=None,
):
    annotated_frame = frame.copy()
    annotated_frame = sv.draw_polygon(
        annotated_frame, polygon=source_polygon, color=sv.Color.GREEN
    )
    annotated_frame = components["trace_annotator"].annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = components["bbox_annotator"].annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = components["label_annotator"].annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    if show_preview:
        # imgResize = cv2.resize(annotated_frame, preview_ratio)
        # cv2.imshow("annotated_frame", imgResize)
        if cv2.waitKey(1) == ord("q"):
            return True
    return annotated_frame


def run_detection(
    SOURCE,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    media_path,
    output_path,
    model_path,
    update_status=None,
    update_preview=None,
    check_input=False,
):
    
    TARGET = np.array(
        [
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ]
    )

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    video_info = sv.VideoInfo.from_video_path(media_path)
    frame_generator = sv.get_video_frames_generator(media_path)
    components = initialize_components(video_info, model_path)
    polygon_zone = sv.PolygonZone(SOURCE)
    views_transformer = ViewsTranformer(SOURCE, TARGET)
    coordinate = defaultdict(lambda: deque(maxlen=video_info.fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    framecount = 0
    conflict_ids = set()
    class_to_ids = defaultdict(set)
    unique_tracker_ids = set()
    # expectedPreviewRatio = (1440, 720)
    expectedPreviewRatio = (1280, 720)
    # showPreview = True
    showPreview = check_input

    conflict_timestamps = []

    for frame in frame_generator:
        if framecount % 5 == 0:
            framecount += 1
            continue

        framecount += 1
        if frame is None:
            break

        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, video_info.fps, (width, height))

        if update_status:
            allFrameCount = video_info.total_frames
            update_status(
                f"Processing frame {framecount}/{allFrameCount}\n{(framecount/allFrameCount)*100:.1f}%"
            )

        result = components["model"](frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        names = result.names
        class_ids = detections.class_id

        detections = detections[polygon_zone.trigger(detections)]
        detections = components["byte_track"].update_with_detections(
            detections=detections
        )

        if detections is None or len(detections) == 0:
            annotated_frame = sv.draw_polygon(
                frame.copy(), polygon=SOURCE, color=sv.Color.GREEN
            )

            out.write(annotated_frame)
            continue

        points = views_transformer.transform_point(
            detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        ).astype(int)
        labels = []

        for i, (tracker_id, [x, y]) in enumerate(zip(detections.tracker_id, points)):
            unique_tracker_ids.add(tracker_id)
            coordinate[tracker_id].append(y)
            class_name = names[class_ids[i]] if i < len(class_ids) else "unknown"
            label = f"[{class_name}] #{tracker_id}"
            # class_to_ids[class_name].add(tracker_id)
            zone = classify_zone((x, y), TARGET_HEIGHT)

            if len(coordinate[tracker_id]) >= video_info.fps / 2:
                d = abs(coordinate[tracker_id][-1] - coordinate[tracker_id][0])
                t = len(coordinate[tracker_id]) / video_info.fps
                speed = d / t * 3.6
                axis = "y" if zone == "A" else "x"
                nearest_distance = find_nearest_object(
                    current_idx=i, points=points, axis=axis
                )

                if nearest_distance is not None and speed > 0:
                    ttc = nearest_distance / speed
                    # if ttc < 2 and tracker_id not in conflict_ids:
                    #     conflict_ids.add(tracker_id)
                    if ttc < 2 and tracker_id not in conflict_ids:
                        conflict_ids.add(tracker_id)
                        timestamp_sec = framecount / video_info.fps
                        class_to_ids[class_name].add(tracker_id)
                        # conflict_timestamps.append(timestamp_sec)
                        # conflict_timestamps.append((timestamp_sec, class_name))
                        conflict_timestamps.append((timestamp_sec, class_name, tracker_id))



                else:
                    ttc = float("inf")

                label = (
                    f"Risk! TTC: {ttc:.2f} s"
                    if ttc < 2
                    else f"[{class_name}] #{tracker_id} {int(speed)} km/h, TTC: {ttc:.2f} s"
                )

            labels.append(label)

        assert len(labels) == len(
            detections
        ), f"Mismatch: {len(labels)} labels for {len(detections)} detections"
        result_frame = annotate_and_write(
            frame,
            detections,
            labels,
            components,
            SOURCE,
            expectedPreviewRatio,
            showPreview,
        )

        if showPreview and update_preview:
            preview_image = Image.fromarray(
                cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            )
            # preview_ratio = (1440, 720)
            preview_ratio = (int(1280 / 2), int(720 / 2))
            preview_image = preview_image.resize(preview_ratio)
            update_preview(preview_image)

        if cv2.waitKey(1) == ord("q") or result_frame is True:
            update_status("Processing complete.")
            break
        out.write(result_frame)

    out.release()
    cv2.destroyAllWindows()

    update_status("Processing complete.")

    summary_path = os.path.splitext(output_path)[0] + "_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Object Detection Summary for: {os.path.basename(media_path)}\n")
        f.write(f"Total unique objects detected: {len(unique_tracker_ids)}\n")
        f.write(f"Total conflicts (TTC < 2s): {len(conflict_ids)}\n\n")
        f.write("Class conflict counts:\n")
        for class_name, id_set in class_to_ids.items():
            f.write(f" - {class_name}: {len(id_set)}\n")

        if conflict_timestamps:
            f.write("\n Conflict Timestamps (in minutes):\n")
            # for i, ts in enumerate(conflict_timestamps):
            #     f.write(f"{i+1} - {ts/60:.2f} minute\n")
            
            # for i, (ts, cls) in enumerate(conflict_timestamps):
            #     f.write(f"{i+1} - {ts/60:.2f} minute ({cls})\n")
            
            for i, (ts, cls, tid) in enumerate(conflict_timestamps):
                f.write(f"{i+1} - {ts/60:.2f} minute ({cls}) [#{tid}]\n")




selected_points = []


def select_points(video_path, source_var):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video file.")
        return

     # ขนาดภาพต้นฉบับ
    original_height, original_width = frame.shape[:2]
    display_width, display_height = 1280, 720
    ratio_x = original_width / display_width
    ratio_y = original_height / display_height
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_points) < 4:
                # selected_points.append((x, y))
                # print(f"Selected Point {len(selected_points)}: ({x}, {y})")
                # แปลงพิกัดจากพรีวิวกลับไปเป็นภาพต้นฉบับ
                actual_x = int(x * ratio_x)
                actual_y = int(y * ratio_y)
                selected_points.append((actual_x, actual_y))
                print(f"Selected Point {len(selected_points)}: ({actual_x}, {actual_y})")
                
            if len(selected_points) == 4:
                cv2.destroyWindow("Select 4 Points")
                source_var.set(str(selected_points))

    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", mouse_callback)

    while True:
        temp_frame = frame.copy()
        for idx, point in enumerate(selected_points):
            cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(
                temp_frame,
                f"{idx+1}",
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        # if len(selected_points) == 4:
        #     cv2.polylines(temp_frame, [np.array(selected_points)], isClosed=True, color=(255, 0, 0), thickness=2)

        if len(selected_points) >= 2:
            pts = np.array(selected_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                temp_frame,
                [pts],
                isClosed=(len(selected_points) == 4),
                color=(255, 0, 0),
                thickness=2,
            )
        # imgResize = cv2.resize(annotated_frame, preview_ratio)

        imgResize = cv2.resize(temp_frame,(1280,720))
        cv2.imshow("Select 4 Points", imgResize)

        if cv2.waitKey(1) & 0xFF == ord("q") or len(selected_points) == 4:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_gui():
    def browse_media():
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        media_var.set(filepath)
        selected_points.clear()
        select_points(filepath, source_var)

    def browse_output():
        filepath = filedialog.asksaveasfilename(defaultextension=".mp4")
        output_var.set(filepath)

    def on_run():
        media = media_var.get()
        mediaOutput = output_var.get()
        target_width = int(width_var.get())
        target_high = int(high_var.get())
        source_str = source_var.get()
        check_input = check_preview_var.get()

        try:
            source = np.array(ast.literal_eval(source_str))
            if source.shape != (4, 2):
                raise ValueError("SOURCE must contain 4 points")
        except Exception as e:
            print(f"Invalid SOURCE: {e}")
            return

        def update_status(text):
            status_var.set(text)
            root.update_idletasks()

        # def update_preview(image):
        #     img_tk = ImageTk.PhotoImage(image=image)
        #     preview_label.imgtk = img_tk
        #     preview_label.config(image=img_tk)
        def update_preview(image):
            img_tk = ImageTk.PhotoImage(image=image)
            preview_label.imgtk = img_tk  # ต้องเก็บ reference
            preview_label.config(image=img_tk)
            preview_label.update()  # Refresh preview label ทันที

        run_detection(
            SOURCE=source,
            TARGET_WIDTH=target_width,
            TARGET_HEIGHT=target_high,
            media_path=media,
            output_path=mediaOutput,
            # model_path="model7.pt",
            model_path=detectModel,
            update_status=update_status,
            # update_preview=update_preview,
            # check_input=check_input
            update_preview=(
                update_preview if check_input else None
            ),  # ถ้าไม่ติ๊ก preview จะไม่ update
            check_input=check_input,
        )

    root = tk.Tk()
    root.title("StopSense")
    root.iconbitmap(default=icon_path)

    preview_label = tk.Label(root)
    preview_label.grid(row=8, column=0, columnspan=3, pady=10)

    status_var = tk.StringVar(value="Waiting...")
    status_label = tk.Label(root, textvariable=status_var, fg="blue")
    status_label.grid(row=6, column=1)

    # checkPreview = tk.BooleanVar()
    # checkPreview = tk.Checkbutton(root, text='Python',textvariable=checkPreview, onvalue=True, offvalue=False)
    # checkPreview.grid(row=7, column=1)
    check_preview_var = tk.BooleanVar(value=False)  # ✅ ตัวแปรควบคุมค่า
    check_preview_checkbutton = tk.Checkbutton(
        root, text="Show Preview", variable=check_preview_var
    )
    check_preview_checkbutton.grid(row=7, column=1)

    tk.Label(root, text="Area points\n(auto-filled after video selection):").grid(
        row=0, column=0
    )
    source_var = tk.StringVar(value="[]")
    tk.Entry(root, textvariable=source_var, width=60).grid(row=0, column=1)

    tk.Label(root, text="Street width:").grid(row=1, column=0)
    width_var = tk.StringVar(value="")
    tk.Entry(root, textvariable=width_var).grid(row=1, column=1)

    tk.Label(root, text="Street length:").grid(row=2, column=0)
    high_var = tk.StringVar(value="")
    tk.Entry(root, textvariable=high_var).grid(row=2, column=1)

    tk.Label(root, text="Input Video:").grid(row=3, column=0)
    media_var = tk.StringVar()
    tk.Entry(root, textvariable=media_var, width=50).grid(row=3, column=1)
    tk.Button(root, text="Browse and Select Points", command=browse_media).grid(
        row=3, column=2
    )

    tk.Label(root, text="Output Video Path:").grid(row=4, column=0)
    output_var = tk.StringVar()
    tk.Entry(root, textvariable=output_var, width=50).grid(row=4, column=1)
    tk.Button(root, text="Browse", command=browse_output).grid(row=4, column=2)

    tk.Button(root, text="Run Detection", command=on_run).grid(row=5, column=1)

    root.mainloop()


def main():
    run_gui()


if __name__ == "__main__":
    main()
