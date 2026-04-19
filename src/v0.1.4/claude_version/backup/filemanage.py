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
    root = tk.Tk()
    root.withdraw() 
    
    file_types = [
        ('Model Files', ('*.pt',)),
        ('All Files', '*.*')
    ]

    model_path = filedialog.askopenfilename(
        title="Select Model File", 
        filetypes=file_types
    )
    
    if model_path:
        print(f"Selected file path: {model_path}")
        return model_path
    else:
        print("No file selected.")
        return None
    
def select_txt_file():
    """
    เปิดหน้าต่าง File Dialog เพื่อให้ผู้ใช้เลือกไฟล์ txt
    และคืนค่าเป็น path ของไฟล์ที่เลือก
    """
    root = tk.Tk()
    root.withdraw() 
    
    file_types = [
        ('text Files', ('*.txt',)),
        ('All Files', '*.*')
    ]

    txt_path = filedialog.askopenfilename(
        title="Select .txt File", 
        filetypes=file_types
    )
    
    if txt_path:
        print(f"Selected file path: {txt_path}")
        return txt_path
    else:
        print("No file selected.")
        return None
    
def select_video_file():
    """
    เปิดหน้าต่าง File Dialog เพื่อให้ผู้ใช้เลือกไฟล์วิดีโอ 
    และคืนค่าเป็น path ของไฟล์ที่เลือก
    """
    root = tk.Tk()
    root.withdraw() 
    
    file_types = [
        ('Video Files', ('*.mp4', '*.avi', '*.mov', '*.mkv')),
        ('All Files', '*.*')
    ]

    video_path = filedialog.askopenfilename(
        title="Select Video File", 
        filetypes=file_types
    )
    
    if video_path:
        print(f"Selected file path: {video_path}")
        return video_path
    else:
        print("No file selected.")
        return None

if __name__ == '__main__':
    selected_path = select_video_file()
    
    if selected_path:
        print(f"\nPath to be used in script: {selected_path}")
    else:
        print("\nOperation cancelled.")

def create_outputfolder(run_id):
    path = f'output/{run_id}'
    os.makedirs(path, exist_ok=True)
    return path

def get_runid():
    date, time = str(datetime.datetime.now()).split(".")[0].split()
    yyyy, mm, dd = date.split("-")
    hh, mu, ss = time.split(":")
    return f"{yyyy}{mm}{dd}_{hh}{mu}{ss}"


def handle_risk_event(
    annotated_frame,
    output_dir,
    frame_number,
    tracker_id_a,
    tracker_id_b,
    class_name_a,
    class_name_b,
    event_type,       # "TTC_RISK" | "PET_RISK"
    metric_value,     # ค่า TTC (วินาที) หรือ PET (วินาที)
    zone_name=None,
    save_log=False,
    save_frame=False,
):
    """
    จัดการเมื่อเกิด risk event ระหว่างวัตถุคู่หนึ่ง

    Parameters
    ----------
    annotated_frame  : np.ndarray  — frame ที่มี annotation แล้ว (สำหรับบันทึก)
    output_dir       : str         — โฟลเดอร์ output
    frame_number     : int         — frame ปัจจุบัน
    tracker_id_a/b   : int         — tracker ID ของวัตถุคู่
    class_name_a/b   : str         — ชื่อ class ของวัตถุคู่
    event_type       : str         — "TTC_RISK" หรือ "PET_RISK"
    metric_value     : float       — ค่า TTC หรือ PET เป็นวินาที
    zone_name        : str | None  — ชื่อ zone ที่เกิดเหตุ
    save_log         : bool        — บันทึก log ลงไฟล์ .txt
    save_frame       : bool        — บันทึก frame เป็น .jpg
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    zone_str = zone_name if zone_name else "unknown"
    metric_label = "TTC" if event_type == "TTC_RISK" else "PET"

    log_line = (
        f"[{timestamp}] | EVENT_TYPE={event_type} | "
        f"pair=(#{tracker_id_a},#{tracker_id_b}) | "
        f"class=({class_name_a},{class_name_b}) | "
        f"{metric_label}={metric_value:.2f}s | "
        f"zone={zone_str} | "
        f"frame={frame_number}"
    )

    # === บันทึก frame ===
    if save_frame and output_dir:
        frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        filename = f"{event_type}_frame{frame_number}_#{tracker_id_a}_#{tracker_id_b}.jpg"
        cv2.imwrite(os.path.join(frame_dir, filename), annotated_frame)

    # === บันทึก log ===
    if save_log and output_dir:
        log_path = os.path.join(output_dir, "risk_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    # === พิมพ์ console ===
    print(log_line)