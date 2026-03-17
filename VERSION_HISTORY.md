# Version History — Collision Risk Detector

บันทึกประวัติการพัฒนาแต่ละ version เพื่อให้กลับมาอ่านทีหลังว่าทำอะไรไปบ้าง

---

## ไฟล์ในโปรเจกต์

| ไฟล์ | คำอธิบาย |
|------|----------|
| `video_homogrphy.py` | ต้นแบบแรก (มี typo ในชื่อ) |
| `video_homography_0-1-1.py` | ✅ v0.1.1 — Base stable version |
| `video_homography_0-1-1_test.py` | v0.1.1 + performance profiling |
| `video_homography_0-1-2.py` | ❌ v0.1.2 — มี bug เส้นชี้ผิดทิศ |
| `video_homography_0-1-2-fix.py` | ✅ v0.1.2-fix — แก้ bug แล้ว (version ล่าสุด) |
| `debug_logger.py` | เครื่องมือ debug สำหรับวิเคราะห์ bug |

---

## v0.1.1 — Base Version
**File:** `video_homography_0-1-1.py`  
**Status:** ✅ Stable

### ฟีเจอร์หลัก:
- **Object Detection** — YOLOv8 custom model (5 classes: Bike, Cars, Human, Truck, Van)
- **Object Tracking** — ByteTrack ผ่าน supervision library
- **Camera Calibration** — เลือก 4 จุดได้ 3 วิธี (mouse, txt file, keyboard input)
- **Homography Transformation** — แปลง image space → real-world space (เมตร)
- **Speed Estimation** — คำนวณความเร็วจริงผ่าน homography
  - `speed_memory` (deque, maxlen=5) เก็บจุดใน zone
  - ใช้ median smoothing เมื่อมีข้อมูล ≥3 ค่า
  - IQR outlier filtering เมื่อวัตถุออกจาก zone
- **Zone Detection** — ตรวจสอบว่าวัตถุอยู่ใน calibration polygon
- **Trace Annotation** — วาดเส้นทางเคลื่อนที่ (trace_length = FPS×2)

### ข้อสังเกต:
- `REAL_FPS` hardcode = 55 (ไม่ได้ดึงจากวิดีโอ)
- ไม่มี prediction line / trajectory prediction

---

## v0.1.1-test — Performance Profiling
**File:** `video_homography_0-1-1_test.py`  
**Status:** 🧪 Testing only

### เพิ่มจาก v0.1.1:
- เพิ่ม `import time` สำหรับ profiling
- วัดเวลาแต่ละขั้นตอน:
  - `Read` — อ่าน frame
  - `YOLO` — inference
  - `Plot` — annotation
  - `Logic` — speed calculation
- print ค่าเวลาทุก frame: `Read:Xms | YOLO:Xms | Plot:Xms | Logic:Xms`

### ข้อสังเกต:
- `REAL_FPS` hardcode = 55 (เหมือน v0.1.1)
- ไม่มีการ import `get_fps`

---

## v0.1.2 — Trajectory Prediction (มี Bug)
**File:** `video_homography_0-1-2.py`  
**Status:** ❌ มี Bug

### เพิ่มจาก v0.1.1:
- **`direction_memory`** (deque, maxlen=20) — เก็บ track points สำหรับคำนวณทิศทาง
- **`estimate_direction()`** — คำนวณ direction vector ด้วย weighted average displacement
- **`project_future_point()`** — คาดคะเนจุดอนาคตผ่าน homography
- **`draw_direction_line()`** — วาดเส้นทำนายทิศทาง (สีแดง)
- **`is_near_frame_edge()`** — ไม่วาดเส้นเมื่อใกล้ขอบ frame (margin=50px)
- เปลี่ยน `REAL_FPS` จาก hardcode → ใช้ `get_fps.get_video_fps_cv2()`
- `PROCESS_EVERY_N_FRAME = 2` (ลดความถี่)

### Bug ที่พบ:
> เส้นทำนายทิศทางบางครั้งชี้ไปด้านหลังแทนที่จะชี้ไปข้างหน้า

**สาเหตุที่คิดตอนแรก** (ถูกบางส่วน):
1. `estimate_direction()` — weighted average อาจให้ direction ที่สวนทาง
2. `project_future_point()` — มี direction check ที่ถูก comment ไว้ (ไม่ได้ใช้งาน)
3. Homography transformation อาจกลับทิศ

---

## v0.1.2-fix — Bug Fix + Debug Features
**File:** `video_homography_0-1-2-fix.py`  
**Status:** ✅ Stable (version ล่าสุด)

### Fix ที่ 1: Dot product check (จาก v0.1.1 analysis)
**ปัญหา:** direction vector อาจชี้สวนทาง  
**แก้ไข:**
- **`estimate_direction()`** — เพิ่ม dot product check เทียบกับ overall displacement (first→last point)
  - ถ้า dot product < 0 → กลับทิศ (vx, vy = -vx, -vy)
- **`project_future_point()`** — เปิด dot product check ใน real-world space
  - เช็คเฉพาะเมื่อ displacement > 0.1 เมตร (ป้องกัน jitter)

### Fix ที่ 2: Homography explosion (จาก debug log analysis)
**ปัญหา:** เมื่อวัตถุออกจาก calibration zone → homography projection ระเบิด (future_point ค่าหลักหมื่น-แสน)  
**สาเหตุจริง:** Homography matrix ทำงานถูกต้องเฉพาะใน calibration polygon เท่านั้น จุดนอก polygon → `cv2.perspectiveTransform()` ให้ค่าที่ผิดเพี้ยน → `H_inv` แปลงกลับเป็น pixel ที่ระเบิด  
**หลักฐาน:** debug log แสดง future_point เปลี่ยนจาก (919,623) → (-214684, 94372) เมื่อออกจาก zone  
**แก้ไข:**
1. **Main loop** — ส่ง `homography_matrix=None` เมื่อ `in_zone=False` → ใช้ fallback (pixel-space direction, 60px)
   ```python
   use_homography = HOMOGRAPHY_MATRIX if in_zone else None
   ```
2. **`project_future_point()`** — เพิ่ม sanity check: ถ้า future_point ไกลเกิน 500px → return None → fallback
   ```python
   MAX_PIXEL_DISTANCE = 500
   if pixel_dist > MAX_PIXEL_DISTANCE:
       return None
   ```

### ฟีเจอร์เพิ่มเติม:
- **`SHOW_FRAME_INFO = True`** — แสดง frame number + เวลา (MM:SS.ms) มุมซ้ายบน
- เปลี่ยน `PROCESS_EVERY_N_FRAME` กลับเป็น `1` (จาก 2 ใน v0.1.2)

---

## Debug Logger
**File:** `debug_logger.py`  
**Status:** 🔧 Debug tool

### วัตถุประสงค์:
สร้างขึ้นเพื่อวิเคราะห์ bug เส้นทำนายทิศทาง เมื่อ fix ที่ 1 ยังไม่หาย

### ฟีเจอร์:
- **เลือก class** — ดึง class list จาก `model.names` แสดงให้ user เลือก
- **เลือก frame range** — by frame number / time range / all frames
- **เก็บ diagnostic data ละเอียดทุก frame ต่อ tracker:**
  - Position, in_zone, near_edge, speed
  - Direction vector (vx, vy), overall displacement
  - `dot_product_image` + `direction_flipped_image`
  - `dot_product_real` + `direction_flipped_real`
  - future_point, prediction_method
  - track_points (JSON only)
- **Export:**
  - `debug_log.csv` — ดูภาพรวมใน Excel/pandas
  - `debug_log.json` — ข้อมูลเต็มรวม track_points array

### ผลจากการใช้:
ค้นพบว่า **bug ไม่ได้เกิดจาก direction flip** (flipped = False ทุก record) แต่เกิดจาก **homography projection ระเบิดนอก calibration zone** → นำไปสู่ Fix ที่ 2

---

## Timeline

```
v0.1.1          Base version (detection + tracking + speed)
  │
  ├─ v0.1.1-test   + performance profiling
  │
  └─ v0.1.2        + prediction lines (มี bug!)
       │
       └─ v0.1.2-fix  Fix 1: dot product checks
              │
              ├─ debug_logger   วิเคราะห์ bug → พบ root cause
              │
              └─ v0.1.2-fix     Fix 2: homography zone check + sanity clamp ✅
```
