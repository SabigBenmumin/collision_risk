import tkinter as tk
from tkinter import filedialog
import os, datetime
import cv2
import numpy as np

def select_model_file():
    """
    เปิดหน้าต่าง File Dialog เพื่อให้ผู้ใช้เลือกไฟล์ model
    และคืนค่าเป็น path ของไฟล์ที่เลือก
    """
    # ซ่อนหน้าต่างหลักของ Tkinter (เพราะเราต้องการแค่ File Dialog)
    root = tk.Tk()
    root.withdraw() 
    
    # กำหนดประเภทไฟล์ที่อนุญาตให้เลือก
    file_types = [
        ('Model Files', ('*.pt',)),
        ('All Files', '*.*')
    ]

    # เปิดหน้าต่างสำหรับเลือกไฟล์
    model_path = filedialog.askopenfilename(
        title="Select Model File", 
        filetypes=file_types
    )
    
    # ตรวจสอบว่าผู้ใช้เลือกไฟล์หรือไม่
    if model_path:
        print(f"Selected file path: {model_path}")
        return model_path
    else:
        print("No file selected.")
        return None

def select_video_file():
    """
    เปิดหน้าต่าง File Dialog เพื่อให้ผู้ใช้เลือกไฟล์วิดีโอ 
    และคืนค่าเป็น path ของไฟล์ที่เลือก
    """
    # ซ่อนหน้าต่างหลักของ Tkinter (เพราะเราต้องการแค่ File Dialog)
    root = tk.Tk()
    root.withdraw() 
    
    # กำหนดประเภทไฟล์ที่อนุญาตให้เลือก
    # 'Video Files' คือชื่อที่จะแสดงใน Dropdown
    # ['*.mp4', '*.avi', '*.mov'] คือนามสกุลไฟล์ที่อนุญาต
    file_types = [
        ('Video Files', ('*.mp4', '*.avi', '*.mov', '*.mkv')),
        ('All Files', '*.*')
    ]

    # เปิดหน้าต่างสำหรับเลือกไฟล์
    video_path = filedialog.askopenfilename(
        title="Select Video File", 
        filetypes=file_types
    )
    
    # ตรวจสอบว่าผู้ใช้เลือกไฟล์หรือไม่
    if video_path:
        print(f"Selected file path: {video_path}")
        return video_path
    else:
        print("No file selected.")
        return None
# ----------------- ตัวอย่างการใช้งาน -----------------

if __name__ == '__main__':
    selected_path = select_video_file()
    
    if selected_path:
        # สามารถนำ selected_path ไปใช้กับโค้ด OpenCV/YOLO ต่อไปได้
        # ตัวอย่างเช่น:
        # cap = cv2.VideoCapture(selected_path) 
        
        print(f"\nPath to be used in script: {selected_path}")
    else:
        print("\nOperation cancelled.")

def create_outputfolder(run_id):
    path = f'ouput/{run_id}'
    # os.path.exists(path)
    os.makedirs(path, exist_ok=True)
    return path

def get_runid():
    date, time = str(datetime.datetime.now()).split(".")[0].split()
    yyyy, mm, dd = date.split("-")
    hh, mu, ss = time.split(":")
    return f"{yyyy}{mm}{dd}_{hh}{mu}{ss}"

import datetime

def handle_risk_event(
    annotated_frame,
    output_dir,
    detections,
    result,
    tracker_id,
    points,
    i,
    cap,
    save_log=True,
    save_frame=True,
):
    """
    ฟังก์ชันจัดการเมื่อเกิด risk event
    - save_log: บันทึกข้อความ log (.txt)
    - save_frame: บันทึกภาพ frame ที่เกิด risk
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # === หา class ของวัตถุหลัก ===
    class_id = detections.class_id[i]
    class_name = result.names[class_id] if class_id < len(result.names) else "unknown"

    # === หา object ที่อยู่ใกล้สุด (คู่เสี่ยงชน) ===
    nearest_idx = np.argmin([
        np.hypot(p[0] - points[i][0], p[1] - points[i][1]) if j != i else np.inf
        for j, p in enumerate(points)
    ])
    nearest_class_id = detections.class_id[nearest_idx]
    nearest_class_name = result.names[nearest_class_id] if nearest_class_id < len(result.names) else "unknown"

    # === (1) Save frame ถ้าต้องการ ===
    if save_frame:
        frame_filename = f"frame_{frame_number}_risk.jpg"
        frame_output_dir = os.path.join(output_dir,r"frame")

        os.makedirs(frame_output_dir, exist_ok=True)

        frame_path = os.path.join(frame_output_dir, frame_filename)
        cv2.imwrite(frame_path, annotated_frame)

    # === (2) Save log ถ้าต้องการ ===
    if save_log:
        log_path = os.path.join(output_dir, "risk_log.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] RISK frame={frame_number} #{tracker_id}: {class_name} vs {nearest_class_name}\n")

    # === พิมพ์ข้อความเพื่อ debug บน console ===
    print(f"[{timestamp}] RISK frame={frame_number} #{tracker_id}: {class_name} vs {nearest_class_name}")
