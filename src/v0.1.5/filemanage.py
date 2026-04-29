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
    
    root.destroy()
    
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
    
    root.destroy()
    
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
    
    root.destroy()
    
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
    collision_type=None,  # "intersection" | "following" | "head_on" (TTC only)
    save_log=False,
    save_frame=False,
    video_fps=None,   # FPS ของวิดีโอ (สำหรับคำนวณเวลาจริง)
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
    collision_type   : str | None  — ประเภทการชน: "intersection", "following", "head_on"
    save_log         : bool        — บันทึก log ลงไฟล์ .txt
    save_frame       : bool        — บันทึก frame เป็น .jpg
    video_fps        : float | None — FPS ของวิดีโอ (สำหรับคำนวณเวลาจริง)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    zone_str = zone_name if zone_name else "unknown"
    metric_label = "TTC" if event_type == "TTC_RISK" else "PET"
    collision_str = f" | type={collision_type}" if collision_type else ""

    # คำนวณเวลาวิดีโอ (นาที:วินาที) เป็น timestamp
    video_time_str = "unknown"
    if video_fps and video_fps > 0:
        time_sec = frame_number / video_fps
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        video_time_str = f"{minutes:02d}:{seconds:05.2f}"

    log_line = (
        f"[{video_time_str}] | EVENT_TYPE={event_type} | "
        f"pair=(#{tracker_id_a},#{tracker_id_b}) | "
        f"class=({class_name_a},{class_name_b}) | "
        f"{metric_label}={metric_value:.2f}s{collision_str} | "
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


def generate_summary_report(output_dir, setup_config=None, input_info=None):
    """
    สร้างไฟล์สรุปการรัน (summary report) จาก risk_log.txt พร้อมข้อมูล setup และ input

    Parameters
    ----------
    output_dir : str — โฟลเดอร์ output ที่มี risk_log.txt
    setup_config : dict | None — ข้อมูล config ที่ใช้ (เช่น TTC_THRESHOLD, REAL_FPS, etc.)
    input_info : dict | None — ข้อมูล input (เช่น video_path, model_path, num_zones)
    """
    if not output_dir:
        return

    log_path = os.path.join(output_dir, "risk_log.txt")
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return

    try:
        # อ่าน log file
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            print("Warning: risk_log.txt is empty")
            return

        # ตัวแปรสำหรับเก็บสถิติ
        ttc_events = []
        pet_events = []
        zone_counts = {}
        collision_type_counts = {}

        # วิเคราะห์แต่ละ line
        for line in lines:
            if not line.strip():
                continue

            parts = line.split(" | ")
            event_data = {}

            for part in parts:
                if "=" in part:
                    key, val = part.split("=", 1)
                    event_data[key.strip()] = val.strip()

            event_type = event_data.get("EVENT_TYPE", "")
            pair_info = event_data.get("pair", "")
            zone_info = event_data.get("zone", "")
            metric_str = event_data.get("TTC", event_data.get("PET", "0"))
            class_info = event_data.get("class", "")
            collision_info = event_data.get("type", "")

            try:
                metric_value = float(metric_str.replace("s", "").strip())
            except:
                metric_value = 0.0

            # เก็บข้อมูล event
            if event_type == "TTC_RISK":
                ttc_events.append({
                    "metric": metric_value,
                    "pair": pair_info,
                    "zone": zone_info,
                    "class": class_info,
                    "type": collision_info,
                })
            elif event_type == "PET_RISK":
                pet_events.append({
                    "metric": metric_value,
                    "pair": pair_info,
                    "zone": zone_info,
                    "class": class_info,
                })

            # Count zones
            if zone_info:
                zone_counts[zone_info] = zone_counts.get(zone_info, 0) + 1

            # Count collision types
            if collision_info:
                collision_type_counts[collision_info] = collision_type_counts.get(collision_info, 0) + 1

        # สร้าง summary report
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("COLLISION RISK DETECTION SUMMARY REPORT")
        summary_lines.append("=" * 80)
        summary_lines.append("")

        # === INPUT INFORMATION ===
        if input_info:
            summary_lines.append("INPUT INFORMATION:")
            summary_lines.append("-" * 80)
            if "video_path" in input_info:
                summary_lines.append(f"Video File: {input_info['video_path']}")
            if "model_path" in input_info:
                summary_lines.append(f"Model File: {input_info['model_path']}")
            if "num_zones" in input_info:
                summary_lines.append(f"Number of Zones: {input_info['num_zones']}")
            summary_lines.append("")

        # === SETUP CONFIGURATION ===
        if setup_config:
            summary_lines.append("SETUP CONFIGURATION:")
            summary_lines.append("-" * 80)
            if "REAL_FPS" in setup_config:
                summary_lines.append(f"Video FPS: {setup_config['REAL_FPS']:.2f}")
            if "TTC_THRESHOLD" in setup_config:
                summary_lines.append(f"TTC Threshold: {setup_config['TTC_THRESHOLD']:.2f}s")
            if "ARRIVAL_GAP" in setup_config:
                summary_lines.append(f"Arrival Gap: {setup_config['ARRIVAL_GAP']:.2f}s")
            if "PET_THRESHOLD" in setup_config:
                summary_lines.append(f"PET Threshold: {setup_config['PET_THRESHOLD']:.2f}s")
            if "RISK_COOLDOWN_S" in setup_config:
                summary_lines.append(f"Risk Cooldown: {setup_config['RISK_COOLDOWN_S']:.2f}s")
            if "TTC_LOOKAHEAD_S" in setup_config:
                summary_lines.append(f"TTC Lookahead: {setup_config['TTC_LOOKAHEAD_S']:.2f}s")
            if "LATERAL_OFFSET_MAX" in setup_config:
                summary_lines.append(f"Lateral Offset Max: {setup_config['LATERAL_OFFSET_MAX']:.2f}m")
            if "DOT_FOLLOWING_MIN" in setup_config:
                summary_lines.append(f"DOT Following Min: {setup_config['DOT_FOLLOWING_MIN']:.2f}")
            if "DOT_HEADON_MAX" in setup_config:
                summary_lines.append(f"DOT Head-on Max: {setup_config['DOT_HEADON_MAX']:.2f}")
            if "RISK_ZONE_MODE" in setup_config:
                summary_lines.append(f"Risk Zone Mode: {setup_config['RISK_ZONE_MODE']}")
            summary_lines.append("")

        # Overall statistics
        total_events = len(ttc_events) + len(pet_events)
        summary_lines.append("=" * 80)
        summary_lines.append("RISK DETECTION RESULTS:")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Total Risk Events: {total_events}")
        summary_lines.append(f"  - TTC Risk Events: {len(ttc_events)}")
        summary_lines.append(f"  - PET Risk Events: {len(pet_events)}")
        summary_lines.append("")

        # TTC Statistics
        if ttc_events:
            ttc_values = [e["metric"] for e in ttc_events]
            summary_lines.append("TTC Risk Statistics (in seconds):")
            summary_lines.append(f"  - Minimum: {min(ttc_values):.2f}s")
            summary_lines.append(f"  - Maximum: {max(ttc_values):.2f}s")
            summary_lines.append(f"  - Average: {sum(ttc_values) / len(ttc_values):.2f}s")
            summary_lines.append("")

        # PET Statistics
        if pet_events:
            pet_values = [e["metric"] for e in pet_events]
            summary_lines.append("PET Risk Statistics (in seconds):")
            summary_lines.append(f"  - Minimum: {min(pet_values):.2f}s")
            summary_lines.append(f"  - Maximum: {max(pet_values):.2f}s")
            summary_lines.append(f"  - Average: {sum(pet_values) / len(pet_values):.2f}s")
            summary_lines.append("")

        # Collision Types (TTC only)
        if collision_type_counts:
            summary_lines.append("Collision Types (TTC Events):")
            for ctype, count in sorted(collision_type_counts.items(), key=lambda x: x[1], reverse=True):
                summary_lines.append(f"  - {ctype}: {count}")
            summary_lines.append("")

        # Zone Distribution
        if zone_counts:
            summary_lines.append("Risk Events by Zone:")
            for zone, count in sorted(zone_counts.items(), key=lambda x: x[1], reverse=True):
                summary_lines.append(f"  - {zone}: {count}")
            summary_lines.append("")

        summary_lines.append("=" * 80)

        # บันทึก summary file
        summary_path = os.path.join(output_dir, "summary_report.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        # พิมพ์ console
        print("\n" + "\n".join(summary_lines))
        print(f"\nSummary report saved to: {summary_path}")

    except Exception as e:
        print(f"Error generating summary report: {e}")