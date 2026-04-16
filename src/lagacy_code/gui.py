import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import threading
import numpy as np
from collections import defaultdict, deque
import os

class CalibrationTool:
    def __init__(self):
        self.points = []
        self.frame = None
        self.original_frame = None
        self.pixel_to_meter = None
        
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
                print("\nReseted - Start selecting a new point")
            elif key == ord('c') and len(self.points) == 4:
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


class YOLOSpeedDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Vehicle Speed Detection & Calibration")
        self.root.geometry("900x700")
        
        # ตัวแปร
        self.video_path = None
        self.pixel_to_meter = None
        self.is_processing = False
        self.is_paused = False
        self.cap = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Menu Bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select Video", command=self.select_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Main Container
        main_container = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top Frame - Control Panel
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Video Selection
        video_frame = ttk.Frame(control_frame)
        video_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(video_frame, text="📁 Select Video File", 
                  command=self.select_video, width=20).pack(side=tk.LEFT, padx=5)
        self.video_label = ttk.Label(video_frame, text="No video selected", 
                                     foreground="gray")
        self.video_label.pack(side=tk.LEFT, padx=10)
        
        # Calibration
        calib_frame = ttk.Frame(control_frame)
        calib_frame.pack(fill=tk.X, pady=5)
        
        self.calib_btn = ttk.Button(calib_frame, text="🎯 Start Calibration", 
                                     command=self.start_calibration, 
                                     state=tk.DISABLED, width=20)
        self.calib_btn.pack(side=tk.LEFT, padx=5)
        
        self.pixel_to_meter_label = ttk.Label(calib_frame, 
                                              text="PIXEL_TO_METER: Not calibrated", 
                                              foreground="red",
                                              font=('Arial', 9, 'bold'))
        self.pixel_to_meter_label.pack(side=tk.LEFT, padx=10)
        
        # Processing
        process_frame = ttk.Frame(control_frame)
        process_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(process_frame, text="▶ Start Detection", 
                                    command=self.start_detection, 
                                    state=tk.DISABLED, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(process_frame, text="⏸ Pause", 
                                    command=self.pause_detection, 
                                    state=tk.DISABLED, width=15)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(process_frame, text="⏹ Stop", 
                                   command=self.stop_detection, 
                                   state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # FPS
        fps_frame = ttk.Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        ttk.Label(fps_frame, text="FPS:", width=20).pack(side=tk.LEFT)
        self.fps_var = tk.IntVar(value=25)
        ttk.Spinbox(fps_frame, from_=1, to=60, textvariable=self.fps_var, 
                    width=10).pack(side=tk.LEFT)
        ttk.Label(fps_frame, text="(frames per second)").pack(side=tk.LEFT, padx=5)
        
        # Confidence
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, pady=2)
        ttk.Label(conf_frame, text="Confidence:", width=20).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.65)
        conf_scale = ttk.Scale(conf_frame, from_=0.0, to=1.0, 
                              variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.pack(side=tk.LEFT, padx=5)
        self.conf_label = ttk.Label(conf_frame, text="0.65", width=8)
        self.conf_label.pack(side=tk.LEFT)
        conf_scale.config(command=lambda v: self.conf_label.config(text=f"{float(v):.2f}"))
        
        # Trace Length
        trace_frame = ttk.Frame(settings_frame)
        trace_frame.pack(fill=tk.X, pady=2)
        ttk.Label(trace_frame, text="Trace Length (seconds):", width=20).pack(side=tk.LEFT)
        self.trace_var = tk.IntVar(value=2)
        ttk.Scale(trace_frame, from_=1, to=5, variable=self.trace_var, 
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=5)
        ttk.Label(trace_frame, textvariable=self.trace_var, width=8).pack(side=tk.LEFT)
        
        # Speed Memory Frames
        speed_frame = ttk.Frame(settings_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(speed_frame, text="Speed Memory Frames:", width=20).pack(side=tk.LEFT)
        self.speed_memory_var = tk.IntVar(value=5)
        ttk.Spinbox(speed_frame, from_=2, to=20, textvariable=self.speed_memory_var, 
                    width=10).pack(side=tk.LEFT)
        ttk.Label(speed_frame, text="(frames to calculate speed)").pack(side=tk.LEFT, padx=5)
        
        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        ttk.Label(status_grid, text="Status:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, padx=5)
        self.status_label = ttk.Label(status_grid, text="Idle", foreground="blue")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_grid, text="Frames Processed:", font=('Arial', 9, 'bold')).grid(
            row=1, column=0, sticky=tk.W, padx=5)
        self.frame_count_label = ttk.Label(status_grid, text="0")
        self.frame_count_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_grid, text="Objects Detected:", font=('Arial', 9, 'bold')).grid(
            row=0, column=2, sticky=tk.W, padx=20)
        self.object_count_label = ttk.Label(status_grid, text="0")
        self.object_count_label.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Log Frame
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, 
                                                  state=tk.DISABLED, wrap=tk.WORD,
                                                  font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # เพิ่ม tag สำหรับสี
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('success', foreground='green')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('info', foreground='blue')
        
    def log(self, message, level='info'):
        """เพิ่มข้อความใน log โดยมีสีตาม level"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = ""
        self.log_text.insert(tk.END, f"{timestamp}{message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.video_label.config(text=filename, foreground="green")
            self.calib_btn.config(state=tk.NORMAL)
            self.log(f"✓ Selected video: {filename}", 'success')
            
            # Reset calibration if already done
            if self.pixel_to_meter:
                self.pixel_to_meter = None
                self.pixel_to_meter_label.config(
                    text="PIXEL_TO_METER: Not calibrated", 
                    foreground="red"
                )
                self.start_btn.config(state=tk.DISABLED)
                self.log("⚠ New video selected. Please recalibrate.", 'warning')
            
    def start_calibration(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "กรุณาเลือกไฟล์วิดีโอก่อน!")
            return
        
        self.log("Starting calibration tool...", 'info')
        self.log("Follow instructions in console and OpenCV window", 'info')
        
        # รัน calibration ใน thread แยก
        thread = threading.Thread(target=self.run_calibration_tool, daemon=True)
        thread.start()
        
    def run_calibration_tool(self):
        try:
            calibrator = CalibrationTool()
            pixel_to_meter = calibrator.run(self.video_path)
            
            if pixel_to_meter:
                self.pixel_to_meter = pixel_to_meter
                # Update UI ใน main thread
                self.root.after(0, self.on_calibration_complete, pixel_to_meter)
            else:
                self.root.after(0, self.log, "✗ Calibration cancelled", 'warning')
                
        except Exception as e:
            self.root.after(0, self.log, f"✗ Calibration error: {str(e)}", 'error')
    
    def on_calibration_complete(self, pixel_to_meter):
        self.pixel_to_meter_label.config(
            text=f"PIXEL_TO_METER: {pixel_to_meter:.5f}", 
            foreground="green"
        )
        self.start_btn.config(state=tk.NORMAL)
        self.log(f"✓ Calibration completed: {pixel_to_meter:.5f} m/px", 'success')
        messagebox.showinfo("Success", 
                           f"Calibration completed!\n\nPIXEL_TO_METER = {pixel_to_meter:.5f}")
        
    def start_detection(self):
        if not self.video_path or self.pixel_to_meter is None:
            messagebox.showwarning("Warning", "กรุณาเลือกวิดีโอและ calibrate ก่อน!")
            return
        
        self.log("=" * 50, 'info')
        self.log("Starting detection...", 'info')
        self.status_label.config(text="Processing", foreground="green")
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.calib_btn.config(state=tk.DISABLED)
        self.is_processing = True
        
        # เริ่ม thread สำหรับประมวลผล
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
        
    def process_video(self):
        try:
            # ตรวจสอบว่ามี YOLO model หรือไม่
            model_path = "models/best.pt"
            if not os.path.exists(model_path):
                self.root.after(0, self.log, f"✗ ERROR: Model file not found at {model_path}", 'error')
                self.root.after(0, messagebox.showerror, "Error", f"ไม่พบไฟล์ model ที่ {model_path}")
                self.root.after(0, self.stop_detection)
                return
            
            # โหลด YOLO (ต้องมี ultralytics และ supervision)
            try:
                from ultralytics import YOLO
                import supervision as sv
            except ImportError as e:
                self.root.after(0, self.log, f"✗ ERROR: Required library not installed - {e}", 'error')
                self.root.after(0, messagebox.showerror, "Error", 
                              "กรุณาติดตั้ง:\npip install ultralytics supervision")
                self.root.after(0, self.stop_detection)
                return
            
            model = YOLO(model_path)
            self.root.after(0, self.log, "✓ Model loaded successfully", 'success')
            
            cap = cv2.VideoCapture(self.video_path)
            fps = self.fps_var.get()
            conf = self.conf_var.get()
            speed_memory_len = self.speed_memory_var.get()
            
            byte_track = sv.ByteTrack(frame_rate=fps)
            trace_annotator = sv.TraceAnnotator(
                trace_length=fps*self.trace_var.get(), 
                thickness=2, 
                position=sv.Position.BOTTOM_CENTER
            )
            
            speed_memory = defaultdict(lambda: deque(maxlen=speed_memory_len))
            frame_count = 0
            total_objects = set()
            
            self.root.after(0, self.log, f"Settings: FPS={fps}, Confidence={conf:.2f}, Speed Memory={speed_memory_len} frames", 'info')
            
            while cap.isOpened() and self.is_processing:
                if self.is_paused:
                    cv2.waitKey(100)
                    continue
                
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                
                # ประมวลผลด้วย YOLO
                results = model.predict(source=frame, stream=True, conf=conf)
                result = next(results)
                
                detections = sv.Detections.from_ultralytics(result)
                detections = byte_track.update_with_detections(detections=detections)
                
                points = detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                ).astype(int)
                
                annotated_frame = trace_annotator.annotate(
                    scene=result.plot(), 
                    detections=detections
                )
                
                # นับจำนวน objects
                if detections.tracker_id is not None:
                    for tid in detections.tracker_id:
                        if tid is not None:
                            total_objects.add(tid)
                
                # คำนวณความเร็ว
                for tracker_id, point in zip(detections.tracker_id, points):
                    if tracker_id is None:
                        continue
                    
                    speed_memory[tracker_id].append(point)
                    
                    if len(speed_memory[tracker_id]) >= 2:
                        p1 = speed_memory[tracker_id][0]
                        p2 = speed_memory[tracker_id][-1]
                        
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        pixel_dist = np.hypot(dx, dy)
                        num_frames = len(speed_memory[tracker_id]) - 1
                        time_elapsed_s = num_frames / fps
                        
                        if time_elapsed_s > 0:
                            speed_mps = (pixel_dist / time_elapsed_s) * self.pixel_to_meter
                            speed_kmh = speed_mps * 3.6
                            
                            cv2.putText(
                                annotated_frame, 
                                f"Speed: {speed_kmh:.1f} km/h",
                                (point[0] - 50, point[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 255, 0), 
                                2
                            )
                
                # Update UI
                if frame_count % 10 == 0:  # Update ทุก 10 frames
                    self.root.after(0, self.frame_count_label.config, 
                                   {'text': str(frame_count)})
                    self.root.after(0, self.object_count_label.config, 
                                   {'text': str(len(total_objects))})
                
                # แสดงผล
                cv2.imshow("YOLOv8 Object Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.root.after(0, self.log, "Detection stopped by user (pressed 'q')", 'info')
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            self.root.after(0, self.log, 
                           f"✓ Processing completed. Total frames: {frame_count}, Objects: {len(total_objects)}", 
                           'success')
            
        except Exception as e:
            self.root.after(0, self.log, f"✗ ERROR: {str(e)}", 'error')
            self.root.after(0, messagebox.showerror, "Error", str(e))
        
        finally:
            self.root.after(0, self.stop_detection)
    
    def pause_detection(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self.status_label.config(text="Paused", foreground="orange")
            self.log("⏸ Detection paused", 'warning')
        else:
            self.pause_btn.config(text="⏸ Pause")
            self.status_label.config(text="Processing", foreground="green")
            self.log("▶ Detection resumed", 'info')
    
    def stop_detection(self):
        self.is_processing = False
        self.is_paused = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self.calib_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Idle", foreground="blue")
        self.log("⏹ Detection stopped", 'info')
        
    def show_instructions(self):
        instructions = """
YOLOv8 Vehicle Speed Detection - คู่มือการใช้งาน

ขั้นตอนการใช้งาน:

1. เลือกวิดีโอ
   - คลิกปุ่ม "Select Video File"
   - เลือกไฟล์วิดีโอที่ต้องการวิเคราะห์

2. Calibration (แปลง Pixel เป็น Meter)
   - คลิกปุ่ม "Start Calibration"
   - จะเปิดหน้าต่าง OpenCV ขึ้นมา
   - คลิก 4 จุดบนพื้นผิวที่รู้ระยะทางจริง ตามลำดับ:
     • บนซ้าย → บนขวา → ล่างขวา → ล่างซ้าย
   - กด 'c' เมื่อคลิกครบ 4 จุด
   - ใส่ระยะทางจริงของแต่ละด้านใน console
   - กด 'q' เพื่อยืนยัน (หรือ 'r' เพื่อเริ่มใหม่)

3. เริ่มตรวจจับ
   - คลิกปุ่ม "Start Detection"
   - ดูผลลัพธ์แบบ real-time
   - กด 'q' บนหน้าต่างวิดีโอเพื่อหยุด

ปุ่มควบคุม:
- Pause/Resume: หยุดชั่วคราว/ดำเนินการต่อ
- Stop: หยุดการทำงาน

Settings:
- FPS: จำนวนเฟรมต่อวินาทีของวิดีโอ
- Confidence: ค่าความมั่นใจในการตรวจจับ (0-1)
- Trace Length: ความยาวของเส้นวิ่งที่แสดง (วินาที)
- Speed Memory: จำนวนเฟรมที่ใช้คำนวณความเร็ว

Tips:
• เลือกบริเวณที่มีขนาดทราบแน่นอนสำหรับ calibration
  เช่น ความกว้างเลนจราจร, ความยาวเส้นจราจร
• ยิ่ง calibration แม่นยำ ผลลัพธ์ยิ่งถูกต้อง
• ใช้ค่า FPS ให้ตรงกับวิดีโอจริง
        """
        messagebox.showinfo("Instructions", instructions)
        
    def show_about(self):
        about_text = """
YOLOv8 Vehicle Speed Detection & Calibration
Version 2.0

โปรแกรมตรวจจับและวัดความเร็วของยานพาหนะ
โดยใช้ YOLOv8 และ ByteTrack

Features:
• Object Detection & Tracking
• Speed Measurement (km/h)
• Camera Calibration
• Real-time Processing

Requirements:
• ultralytics
• supervision
• opencv-python
• numpy

Developed with Python, OpenCV, and Tkinter
© 2024
        """
        messagebox.showinfo("About", about_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOSpeedDetectionApp(root)
    root.mainloop()