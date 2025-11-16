import cv2
import os
from tkinter import filedialog
import tkinter as tk

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

def extract_frames(video_path, output_folder):
    # สร้างโฟลเดอร์สำหรับเก็บภาพ ถ้ายังไม่มี
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    
    # เปิดไฟล์วิดีโอ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    save_frames_count = 0
    while True:
        # อ่านเฟรมถัดไป
        ret, frame = cap.read()
        
        # ถ้า ret เป็น False หมายความว่าสิ้นสุดวิดีโอแล้ว ให้หยุด loop
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        
        if frame_count % (25*30) == 0:
            cv2.imwrite(frame_filename, frame)
            save_frames_count += 1
        
        frame_count += 1

    # ปล่อยทรัพยากร
    cap.release()
    print(f"\nCompleted! Total frames extracted: {save_frames_count}")

# --- ตัวอย่างการใช้งาน ---
# กำหนดเส้นทางไฟล์วิดีโอของคุณ
input_video_file = select_video_file()
video_name = input_video_file.split("/")[-1]
print(video_name)
# กำหนดชื่อโฟลเดอร์สำหรับเก็บรูปภาพที่สกัด
output_dir = 'extracted_frames_dataset'+"/"+ str(video_name).rstrip(".mp4")

# รันฟังก์ชัน
extract_frames(input_video_file, output_dir)
