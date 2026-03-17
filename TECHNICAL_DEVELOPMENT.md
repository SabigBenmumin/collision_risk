# เอกสารเทคนิค: การพัฒนา Collision Risk Detector v0.1.1 → v0.1.2 → v0.1.2-fix

## สารบัญ

1. [ภาพรวมการพัฒนา](#1-ภาพรวมการพัฒนา)
2. [v0.1.1 — สถาปัตยกรรมพื้นฐาน](#2-v011--สถาปัตยกรรมพื้นฐาน)
3. [v0.1.2 — เพิ่ม Trajectory Prediction](#3-v012--เพิ่ม-trajectory-prediction)
4. [Bug Analysis: เส้นชี้ผิดทิศ](#4-bug-analysis-เส้นชี้ผิดทิศ)
5. [v0.1.2-fix — การแก้ไข Bug](#5-v012-fix--การแก้ไข-bug)
6. [Debug Logger — เครื่องมือวิเคราะห์](#6-debug-logger--เครื่องมือวิเคราะห์)
7. [สรุปบทเรียน](#7-สรุปบทเรียน)

---

## 1. ภาพรวมการพัฒนา

```
v0.1.1 (Base)
│  ฟีเจอร์: Detection + Tracking + Speed Estimation
│  ไฟล์: video_homography_0-1-1.py (519 lines)
│
└──▶ v0.1.2 (เพิ่ม Prediction Lines)
    │  ฟีเจอร์ใหม่: estimate_direction, project_future_point, draw_direction_line
    │  ไฟล์: video_homography_0-1-2.py (657 lines, +138 lines)
    │  ❌ มี bug: เส้นทำนายบางครั้งชี้ไปด้านหลัง
    │
    └──▶ v0.1.2-fix (แก้ Bug)
         │  Fix 1: Dot product direction check
         │  Fix 2: Homography zone boundary check
         │  ไฟล์: video_homography_0-1-2-fix.py (695 lines)
         │
         └──▶ debug_logger.py (เครื่องมือวิเคราะห์)
              ฟีเจอร์: Class filter, Frame range, CSV/JSON diagnostic log
              ไฟล์: debug_logger.py (773 lines)
```

---

## 2. v0.1.1 — สถาปัตยกรรมพื้นฐาน

### 2.1 Processing Pipeline

```
Video Frame
    │
    ▼
┌─────────────────┐
│  YOLO Detection  │  model.predict(conf=0.65)
│  (YOLOv8)        │  Output: bounding boxes + class IDs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ByteTrack       │  byte_track.update_with_detections()
│  (Tracking)      │  Output: tracker_id per detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Zone Check      │  is_point_in_polygon(point, CALIBRATION_POLYGON)
│                  │  ตรวจสอบว่า BOTTOM_CENTER อยู่ใน zone
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
 in_zone   out_zone
    │         │
    ▼         ▼
 เก็บ speed   คำนวณ final speed
 memory       (IQR + median)
    │         แล้วล้าง memory
    ▼
 คำนวณ speed
 แบบ real-time
```

### 2.2 Speed Estimation Algorithm

**Data structures:**
```python
speed_memory    = defaultdict(lambda: deque(maxlen=5))   # จุดล่าสุด 5 จุดใน zone
speed_history   = defaultdict(list)                       # ประวัติ speed ทั้งหมดใน zone
last_speed      = defaultdict(float)                      # ความเร็วล่าสุด (km/h)
object_in_zone  = defaultdict(bool)                       # สถานะ in/out zone
```

**ขั้นตอนคำนวณ (เมื่อ in_zone = True):**

```python
# 1. เก็บจุดใน speed_memory
speed_memory[tracker_id].append(point)

# 2. คำนวณ real distance ผ่าน homography
p1 = speed_memory[tracker_id][0]      # จุดแรก
p2 = speed_memory[tracker_id][-1]     # จุดล่าสุด
real_distance = calculate_real_distance(p1, p2, HOMOGRAPHY_MATRIX)
#   ↳ transform_point(p1, H) → real_p1 (เมตร)
#   ↳ transform_point(p2, H) → real_p2 (เมตร)
#   ↳ euclidean_distance(real_p1, real_p2)

# 3. คำนวณ speed
time_elapsed = (len(speed_memory) - 1) / FPS    # วินาที
speed_mps = real_distance / time_elapsed         # m/s
speed_kmh = speed_mps * 3.6                      # km/h

# 4. Smoothing ด้วย median
speed_history[tracker_id].append(speed_kmh)
if len(speed_history) >= 3:
    display_speed = np.median(speed_history[tracker_id])
```

**เมื่อวัตถุออกจาก zone (IQR Outlier Filtering):**

```python
speeds = np.array(speed_history[tracker_id])
Q1 = np.percentile(speeds, 25)
Q3 = np.percentile(speeds, 75)
IQR = Q3 - Q1

# กรองค่าผิดปกติ
valid_range = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
filtered = speeds[(speeds >= valid_range[0]) & (speeds <= valid_range[1])]
final_speed = np.median(filtered)
```

เหตุผลที่ใช้ **median แทน mean**: ทนทานต่อค่าผิดปกติที่เกิดจาก tracking jitter หรือ occlusion ชั่วขณะ ค่า mean จะถูกดึงโดย outlier ทำให้ speed แสดงผิด

### 2.3 Homography Transformation

```
Image Space (pixels)              Real-World Space (meters)
┌─────────────────┐     H         ┌──────────────────┐
│  P1 ──── P2     │  ──────▶     │  (0,0) ── (W,0)  │
│  │        │     │              │  │         │      │
│  │        │     │              │  │         │      │
│  P4 ──── P3     │              │  (0,H) ── (W,H)  │
└─────────────────┘              └──────────────────┘
 (perspective distortion)          (rectangular, meters)
```

```python
H, _ = cv2.findHomography(src_points, dst_points)
# src = 4 จุดใน image (pixels)
# dst = 4 จุดใน real-world (meters) → สี่เหลี่ยมผืนผ้า

# แปลงจุด:
pt = np.array([[[x, y]]], dtype=np.float32)
real_pt = cv2.perspectiveTransform(pt, H)  # → [real_x, real_y] (meters)
```

### 2.4 ข้อสังเกตใน v0.1.1

| Item | ค่า | หมายเหตุ |
|------|-----|----------|
| `REAL_FPS` | hardcode `55` | ❌ ไม่ได้ดึงจากวิดีโอ |
| `PROCESS_EVERY_N_FRAME` | `1` | ประมวลผลทุก frame |
| Prediction line | ไม่มี | ยังไม่ได้พัฒนา |
| Confidence threshold | `0.65` | |

---

## 3. v0.1.2 — เพิ่ม Trajectory Prediction

### 3.1 ฟังก์ชันใหม่ที่เพิ่ม

v0.1.2 เพิ่ม 4 ฟังก์ชันใหม่ (+138 lines):

| ฟังก์ชัน | บรรทัด | หน้าที่ |
|----------|--------|---------|
| `estimate_direction()` | 387-420 | คำนวณ direction vector จาก track points |
| `project_future_point()` | 422-464 | คาดคะเนจุดอนาคตผ่าน homography |
| `draw_direction_line()` | 465-480 | วาดเส้นทำนายทิศทาง |
| `is_near_frame_edge()` | 482-486 | ตรวจสอบว่าใกล้ขอบ frame |

Data structure ใหม่:
```python
direction_memory = defaultdict(lambda: deque(maxlen=20))
# เก็บ track points ล่าสุด 20 จุดสำหรับคำนวณทิศทาง
```

### 3.2 estimate_direction() — Weighted Average Displacement

**Algorithm:**

```
Points: P0 → P1 → P2 → P3 → P4 → ... → Pn
         ↕     ↕     ↕     ↕     ↕
        d1    d2    d3    d4    d5
       (w=1) (w=2) (w=3) (w=4) (w=5)
```

```python
# แต่ละ displacement (di) ระหว่างจุดที่ติดกัน ได้ weight ตามลำดับ
# จุดใหม่กว่า → weight มากกว่า → มีอิทธิพลต่อทิศทางมากกว่า

for i in range(1, n):
    dx = pts[i][0] - pts[i-1][0]
    dy = pts[i][1] - pts[i-1][1]
    w = i                            # weight = index (1, 2, 3, ...)
    total_vx += dx * w
    total_vy += dy * w
    total_w += w

vx = total_vx / total_w             # weighted average x
vy = total_vy / total_w             # weighted average y

# normalize เป็น unit vector
norm = sqrt(vx² + vy²)
vx /= norm
vy /= norm
```

**ทำไมใช้ weighted average แทน cv2.fitLine:**

v0.1.2 มีโค้ด `cv2.fitLine()` ที่ถูก **comment ไว้** (line 366-385):
```python
# output = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
# vx, vy, x0, y0 = float(output[0]), ...
```
`fitLine` ใช้ least squares fit ซึ่งให้ "เส้นที่ fit ดีที่สุด" แต่ **ไม่มี temporal weighting** — จุดเก่าและจุดใหม่มีน้ำหนักเท่ากัน ทำให้ไม่ตอบสนองต่อการเปลี่ยนทิศ

สิ่งที่น่าสนใจคือ `fitLine` version มี **dot product check** อยู่แล้ว:
```python
# if (vx * dx + vy * dy) < 0:
#     vx, vy = -vx, -vy
```
แต่ weighted average version ที่ใช้จริงใน v0.1.2 **ไม่มี check นี้** → เป็น bug แรก

### 3.3 project_future_point() — Homography-Based Projection

**Pipeline:**

```
1. Direction Vector (pixel)      (vx, vy) จาก estimate_direction()
         │
         ▼
2. แปลง direction → real-world   ด้วย homography
   ├─ pt_base = H(x0, y0)
   └─ pt_tip  = H(x0 + vx*100, y0 + vy*100)
   real_direction = normalize(pt_tip - pt_base)
         │
         ▼
3. คำนวณจุดอนาคตใน real-world
   future_real = current_real + real_direction × (speed × time)
         │
         ▼
4. แปลงกลับเป็น pixel            ด้วย H⁻¹
   future_pixel = H⁻¹(future_real)
```

**สิ่งที่ถูก comment ไว้ (Bug ที่ 2):**

```python
# # ตรวจสอบทิศใน real-world space อีกครั้ง
# first_real = transform_point(track_points[0], homography_matrix)
# last_real  = transform_point(track_points[-1], homography_matrix)
# if first_real is not None and last_real is not None:
#     dx = last_real[0] - first_real[0]
#     dy = last_real[1] - first_real[1]
#     if (real_vx * dx + real_vy * dy) < 0:
#         real_vx, real_vy = -real_vx, -real_vy
```

โค้ดนี้เป็น **direction sanity check ใน real-world space** ซึ่งมีความสำคัญมากเพราะ homography transformation อาจกลับทิศ direction vector ได้ แต่ถูก comment ไว้ → ไม่ได้ใช้งาน

### 3.4 draw_direction_line() — Two-Mode Drawing

```python
if speed > 0 and homography is not None:
    # Mode 1: Homography projection
    #   ใช้ speed + direction → คำนวณจุดอนาคตจริง
    #   ความยาวเส้นขึ้นอยู่กับ speed × n_seconds
    future = project_future_point(...)
    cv2.line(frame, current, future, red, 2)
else:
    # Mode 2: Fallback (pixel-space)
    #   ใช้ direction vector ตรงๆ × 60 pixels
    #   ความยาวเส้นคงที่ 60px
    end = current + direction * 60
    cv2.line(frame, current, end, red, 2)
```

### 3.5 การเปลี่ยนแปลง Config

| Item | v0.1.1 | v0.1.2 | เหตุผล |
|------|--------|--------|--------|
| `REAL_FPS` | `55` (hardcode) | `get_fps.get_video_fps_cv2()` | ดึงจากวิดีโอจริง |
| `PROCESS_EVERY_N_FRAME` | `1` | `2` | ลด load (ประมวลผลทุก 2 frame) |
| `direction_memory` | ไม่มี | `deque(maxlen=20)` | เก็บ track points |

### 3.6 Main Loop — Integration

```python
# ใน main loop (ทุก frame, ทุก tracker):
direction_memory[tracker_id].append(point)
angle, dir_vec = estimate_direction(direction_memory[tracker_id])

if dir_vec is not None and not is_near_frame_edge(point, frame.shape):
    current_speed_mps = last_speed[tracker_id] / 3.6
    draw_direction_line(
        frame, point, dir_vec,
        track_points=list(direction_memory[tracker_id]),
        speed_mps=current_speed_mps,
        n_seconds=1,
        homography_matrix=HOMOGRAPHY_MATRIX,   # ← ส่งเสมอ ไม่เช็ค in_zone
    )
```

**ปัญหาที่ซ่อนอยู่:** `HOMOGRAPHY_MATRIX` ถูกส่งไป **ทุก object ไม่ว่าจะอยู่ใน zone หรือไม่** → homography ถูกใช้กับจุดที่อยู่นอก calibration polygon → ค่าระเบิด

---

## 4. Bug Analysis: เส้นชี้ผิดทิศ

### 4.1 อาการที่เห็น

เส้นทำนายทิศทาง (สีแดง) บางครั้ง:
- ชี้ไปด้านหลังของวัตถุ (สวนทางการเคลื่อนที่)
- ยาวมหาศาล (พุ่งออกนอกจอ)
- กระโดดไปทิศสุ่ม

### 4.2 สาเหตุที่คิดตอนแรก (สมมติฐาน)

| # | สมมติฐาน | สถานะ |
|---|---------|-------|
| 1 | Weighted average ใน `estimate_direction()` ให้ direction ที่สวนทาง | ⚠️ เป็นไปได้แต่ไม่ใช่สาเหตุหลัก |
| 2 | Direction check ใน `project_future_point()` ถูก comment ไว้ | ⚠️ เป็นส่วนหนึ่งของปัญหา |
| 3 | Homography transformation กลับทิศ direction vector | ⚠️ เกี่ยวข้อง |

### 4.3 สาเหตุจริง (ค้นพบจาก Debug Log)

**Root Cause: Homography projection ระเบิดเมื่อ object อยู่นอก calibration zone**

```
Calibration Zone (4 จุด)           Homography ทำงาน
┌────────────────────┐
│   P1 ────── P2     │  ← ภายใน polygon นี้ homography ถูกต้อง
│   │          │     │
│   P4 ────── P3     │
└────────────────────┘
         │
     Object เคลื่อนที่ออกจาก zone
         │
         ▼
    Object อยู่นอก polygon
    ↳ transform_point() ได้ค่า real-world ที่เพี้ยน
    ↳ project_future_point() คำนวณจุดอนาคตใน real-world
    ↳ H⁻¹ แปลงกลับ → pixel ระเบิด (หลักหมื่น-แสน)
```

**หลักฐานจาก debug_log.csv:**

```
Tracker #2 (Bike):

Frame 919 | in_zone=True  | point=(1013,591) | future=(919, 623)      ✅ ปกติ
Frame 940 | in_zone=True  | point=(930, 625) | future=(400, 841)      ⚠️ เริ่มไกล
Frame 952 | in_zone=False | point=(866, 654) | future=(-149, 1106)    ❌ ออกนอกจอ
Frame 970 | in_zone=False | point=(707, 724) | future=(-12917, 6647)  💥 ระเบิด
Frame 972 | in_zone=False | point=(681, 735) | future=(-214684,94372) 💥💥 สุดขีด
Frame 973 | in_zone=False | point=(668, 738) | future=(29343, -10978) 💥 กลับทิศ!
```

**Key insight:** `direction_flipped_image` = **False ทุก record** (0/165)  
→ direction vector ชี้ถูกทิศตลอด → ปัญหาไม่ได้อยู่ที่ `estimate_direction()`  
→ ปัญหาอยู่ที่ **homography projection ระเบิดนอก zone** จนเส้นยาวมหาศาลชี้ไปทิศสุ่ม

### 4.4 ทำไม Homography ระเบิด (คณิตศาสตร์)

Homography matrix H แปลงจุดจาก image → real-world:

```
[x']   [h11 h12 h13] [x]
[y'] = [h21 h22 h23] [y]
[w']   [h31 h32 h33] [1]

real_x = x'/w',  real_y = y'/w'
```

เมื่อ `w'` เข้าใกล้ 0 (จุดอยู่ใกล้ "vanishing line" ของ homography):
- `real_x = x'/w'` → **ค่าพุ่งไป ±∞**
- inverse transform `H⁻¹` ก็มีปัญหาเดียวกัน

จุดที่อยู่ภายใน calibration polygon → `w'` อยู่ในช่วงปกติ → transform ถูกต้อง  
จุดที่อยู่นอก polygon โดยเฉพาะไกลจาก polygon → `w'` เข้าใกล้ 0 → **ค่าระเบิด**

---

## 5. v0.1.2-fix — การแก้ไข Bug

### 5.1 Fix 1: Dot Product Direction Check

#### 5.1.1 estimate_direction() — Image Space Check

**ก่อนแก้ (v0.1.2):** ไม่มี direction check

**หลังแก้ (v0.1.2-fix):**

```python
# คำนวณ overall displacement (จุดแรก → จุดสุดท้าย)
overall_dx = float(pts[-1][0] - pts[0][0])
overall_dy = float(pts[-1][1] - pts[0][1])
overall_norm = sqrt(overall_dx² + overall_dy²)

if overall_norm > 1e-6:
    # dot product = |v| × |d| × cos(θ)
    # ถ้า dot < 0 → cos(θ) < 0 → θ > 90° → ชี้สวนทาง!
    dot_product = vx * overall_dx + vy * overall_dy
    if dot_product < 0:
        vx = -vx    # กลับทิศ
        vy = -vy
```

**วิธีทำงาน:**

```
กรณีปกติ (dot > 0):                 กรณี bug (dot < 0):
                                    
  P0 ──▶ Pn                          P0 ──▶ Pn
    overall ──▶                         overall ──▶
    direction ──▶  ✅ ตรงกัน            direction ◀──  ❌ สวนทาง
                                              │
                                              ▼ flip!
                                        direction ──▶  ✅ แก้แล้ว
```

#### 5.1.2 project_future_point() — Real-World Space Check

**ก่อนแก้ (v0.1.2):** มี check แต่ **ถูก comment ไว้**

**หลังแก้ (v0.1.2-fix):** uncomment + เพิ่ม displacement threshold

```python
if len(track_points) >= 2:
    first_real = transform_point(track_points[0], homography_matrix)
    last_real  = transform_point(track_points[-1], homography_matrix)
    
    dx = last_real[0] - first_real[0]
    dy = last_real[1] - first_real[1]
    displacement_norm = sqrt(dx² + dy²)
    
    # ตรวจสอบเฉพาะเมื่อ displacement > 0.1 เมตร
    # (ป้องกัน false flip จาก jitter เมื่อวัตถุแทบไม่เคลื่อนที่)
    if displacement_norm > 0.1:
        if (real_vx * dx + real_vy * dy) < 0:
            real_vx, real_vy = -real_vx, -real_vy
```

**ทำไมต้อง check 2 ระดับ:**

```
1. Image Space Check  →  แก้ปัญหา weighted average ให้ direction ผิดทิศ
                           (เกิดจาก jitter/noise ใน tracking data)

2. Real-World Check   →  แก้ปัญหา homography transformation กลับทิศ
                           (เกิดจาก perspective distortion:
                            direction ที่ถูกใน pixel อาจผิดใน real-world)
```

### 5.2 Fix 2: Homography Zone Boundary Check

**ปัญหา:** Fix 1 ไม่สามารถแก้ปัญหา "homography ระเบิดนอก zone" ได้ เพราะแม้ direction ถูกทิศ แต่การ project จุดอนาคตด้วย H⁻¹ ยังให้ค่าระเบิด

#### 5.2.1 Main Loop — Zone-Based Homography Switch

```python
# ก่อนแก้ (v0.1.2, v0.1.2-fix ก่อน Fix 2):
draw_direction_line(
    ...,
    homography_matrix=HOMOGRAPHY_MATRIX,    # ← ส่งทุกกรณี
)

# หลังแก้:
use_homography = HOMOGRAPHY_MATRIX if in_zone else None   # ← เช็ค zone!
draw_direction_line(
    ...,
    homography_matrix=use_homography,       # ← None เมื่อนอก zone → ใช้ fallback
)
```

**ผล:** เมื่อ `homography_matrix=None` → `draw_direction_line()` จะใช้ fallback mode (เส้น 60px ตาม direction vector ใน pixel-space) ซึ่งไม่มีปัญหา projection ระเบิด

#### 5.2.2 project_future_point() — Sanity Check

เพิ่มเป็น safety net (ป้องกันกรณีที่อยู่ใน zone แต่ใกล้ขอบ):

```python
result = (int(future_pixel[0][0][0]), int(future_pixel[0][0][1]))

# Sanity check: future_point ต้องไม่ไกลจาก current point เกินไป
MAX_PIXEL_DISTANCE = 500  # pixels
dx = result[0] - point[0]
dy = result[1] - point[1]
pixel_dist = sqrt(dx² + dy²)

if pixel_dist > MAX_PIXEL_DISTANCE:
    return None    # ← trigger fallback ใน draw_direction_line()
```

**ทำไมเลือก 500px:**
- At 30 km/h, 1 second prediction ≈ 8.3 เมตร → ในกล้อง CCTV ทั่วไป ≈ 100-300px
- ค่า 500px ให้ margin พอสำหรับ speed สูงและมุมกล้องต่างๆ
- ค่าที่ระเบิด (หลักพัน-หลักแสน) ถูก reject 100%

### 5.3 SHOW_FRAME_INFO — Debug Overlay

```python
# Config (line 13)
SHOW_FRAME_INFO = True

# Overlay (ก่อน cv2.imshow)
if SHOW_FRAME_INFO:
    time_sec = frame_count / REAL_FPS
    minutes = int(time_sec // 60)
    seconds = time_sec % 60
    info_text = f"Frame: {frame_count}  Time: {minutes:02d}:{seconds:05.2f}"
    
    # วาดพื้นหลังดำ + ข้อความขาว มุมซ้ายบน
    (tw, th), _ = cv2.getTextSize(info_text, FONT, 0.6, 2)
    cv2.rectangle(frame, (8,8), (18+tw, 18+th+8), (0,0,0), -1)
    cv2.putText(frame, info_text, (12, 12+th), FONT, 0.6, (255,255,255), 2)
```

### 5.4 สรุปความเปลี่ยนแปลง v0.1.2-fix

| เปลี่ยนแปลง | บรรทัด | ประเภท |
|-------------|--------|--------|
| `SHOW_FRAME_INFO = True` | 13 | ฟีเจอร์ใหม่ |
| Dot product check ใน `estimate_direction()` | 402-413 | Fix 1 |
| Uncomment + threshold ใน `project_future_point()` | 452-465 | Fix 1 |
| `PROCESS_EVERY_N_FRAME = 1` (จาก 2) | 512 | Config change |
| `use_homography = H if in_zone else None` | 589-590 | Fix 2 |
| Sanity check `MAX_PIXEL_DISTANCE = 500` | 476-484 | Fix 2 |
| Frame info overlay | 663-673 | ฟีเจอร์ใหม่ |

---

## 6. Debug Logger — เครื่องมือวิเคราะห์

### 6.1 ทำไมต้องสร้าง

หลัง Fix 1 แล้ว user รายงานว่า bug ยังอยู่ → ต้องมีเครื่องมือเก็บ data ละเอียดเพื่อหาสาเหตุจริง

### 6.2 Diagnostic Data ที่เก็บ

```python
record = {
    # ── Context ──
    "frame_num":     1234,
    "time_sec":      24.680,
    "tracker_id":    5,
    "class_name":    "Bike",
    "point_x":       800,     "point_y":    600,
    "in_zone":       True,
    "near_edge":     False,
    "speed_kmh":     32.5,
    
    # ── Image Space Direction ──
    "vx_raw":        -0.912,  "vy_raw":     0.410,    # ก่อน flip
    "overall_dx":    -100.0,  "overall_dy": 45.0,     # first→last point
    "dot_product_image":     107.3,                    # > 0 = ตรงทิศ
    "direction_flipped_image": False,                   # True = flip แล้ว
    
    # ── Real-World Direction ──
    "real_vx":       -0.068,  "real_vy":    0.998,
    "dot_product_real":      4.55,                     # > 0 = ตรงทิศ
    "direction_flipped_real": False,
    
    # ── Prediction ──
    "prediction_method":  "homography",    # หรือ "fallback" / "none"
    "future_point_x":     650,
    "future_point_y":     720,
    
    # ── Track Data (JSON only) ──
    "track_points":        [[810,595], [805,598], ...],
    "speed_memory_points": [[800,600], [798,602], ...],
}
```

### 6.3 วิธีวิเคราะห์ Bug จาก Log

```
Step 1: เปิด CSV ใน Excel
Step 2: กรอง direction_flipped_image = True
        → ถ้ามี: bug ที่ estimate_direction()
        → ถ้าไม่มี: ดูต่อ Step 3

Step 3: กรอง direction_flipped_real = True  
        → ถ้ามี: bug ที่ homography transformation
        → ถ้าไม่มี: ดูต่อ Step 4

Step 4: ดู future_point ที่ค่าผิดปกติ (|x| > 2000 หรือ |y| > 2000)
        ร่วมกับ in_zone = False
        → ถ้ามี: bug ที่ homography ระเบิดนอก zone ← นี่คือสาเหตุจริง
```

---

## 7. สรุปบทเรียน

### 7.1 Bug ที่ซ่อนตัว

| สิ่งที่เห็น | สิ่งที่คิดว่าเป็นสาเหตุ | สาเหตุจริง |
|-------------|------------------------|-----------|
| เส้นชี้ไปข้างหลัง | direction vector ผิด | homography projection ระเบิด |
| เส้นยาวเกินจอ | speed สูงเกิน | `H⁻¹` ให้ค่า pixel หลักหมื่น-แสน |
| เส้นกระโดดไปทิศสุ่ม | tracking ไม่เสถียร | `w'→0` ใน homography transform |

### 7.2 แนวทางที่ได้เรียนรู้

1. **Homography ไม่ใช่ global transform** — ถูกต้องเฉพาะใน calibration region
2. **ต้องมี sanity check** บน output เสมอ ไม่ใช่แค่ตรวจ input
3. **Debug logging เป็นเครื่องมือที่ทรงพลัง** — ดีกว่าดูด้วยตา ค้นพบ root cause ใน 1 run
4. **Fix แรกอาจไม่ใช่ fix สุดท้าย** — dot product check แก้ได้บางส่วน แต่ root cause อยู่ที่อื่น
