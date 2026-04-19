# Dev Note — Risk Detection Module
**description:** แก้จาก version ที่ commit "add v0.1.4" ไป
**Project:** Traffic Collision Risk Detection System  
**Date:** 2026-04-19  
**Files:** `main.py`, `filemanage.py`

---

## สิ่งที่ทำในรอบนี้

### 1. เพิ่ม Risk Detection Pipeline (TTC + PET)

เพิ่ม collision risk detection แบบ dual-metric เมื่อเกิด event ความเสี่ยงจาก **metric ใด metric หนึ่ง** จะทำการเก็บ log ทันที

---

### 2. TTC (Time to Collision) — Trajectory Intersection Method

**ปัญหาของ TTC แบบ closing speed (วิธีเดิมที่ไม่ใช้แล้ว):**

```
closing_speed = dot(relative_velocity, unit_vec_A→B)
TTC = distance / closing_speed
```

วิธีนี้แค่วัดว่าวัตถุสองตัว "เคลื่อนเข้าหากัน" หรือเปล่า  
→ วิ่งสวนทางกันคนละเลนก็ trigger ได้ เพราะ closing_speed > 0  
→ **ไม่รู้ว่า trajectory จะตัดกันจริงหรือเปล่า = false positive สูงมาก**

**วิธีที่ใช้จริง (trajectory intersection):**

```
1. แปลง direction vector จาก pixel space → real-world space (ผ่าน homography)
2. Project future path ของแต่ละตัวออกไป lookahead_s วินาที
   → segment A_now → A_future (real-world)
   → segment B_now → B_future (real-world)
3. ถ้าสอง segment ไม่ตัดกัน → return None (ไม่เสี่ยง)
4. ถ้าตัดกัน → conflict point C
     TTC_A = dist(A_now, C) / speed_A
     TTC_B = dist(B_now, C) / speed_B
5. เสี่ยงเมื่อ:
     (TTC_A < TTC_THRESHOLD OR TTC_B < TTC_THRESHOLD)   ← ถึงเร็ว
     AND |TTC_A - TTC_B| < ARRIVAL_GAP                  ← ถึงพร้อมกัน
```

**Threshold ที่ตั้งไว้ (ปรับได้ใน config):**

| ตัวแปร | ค่า default | ความหมาย |
|---|---|---|
| `TTC_THRESHOLD` | 3.0 s | ถ้า TTC < นี้ถือว่าถึงเร็ว |
| `ARRIVAL_GAP` | 1.5 s | ช่วงห่างสูงสุดที่ถือว่า "ถึงพร้อมกัน" |
| `TTC_LOOKAHEAD_S` | 4.0 s | ความยาว future path ที่ project ออกไป |

**ฟังก์ชันหลัก:**
- `_get_real_direction_vector()` — แปลง pixel dir_vec → real-world พร้อม dot-product check ทิศทาง
- `_ray_segment_intersection_2d()` — หาจุดตัดของ ray กับ segment ใน 2D
- `compute_ttc()` — logic หลัก คืน `(ttc_a, ttc_b)` หรือ `None`

---

### 3. PET (Post Encroachment Time) — Trace Path Intersection

**แนวคิด:**

TTC ดู future (ก่อนเกิด) แต่ PET ดู trace จริงที่เกิดขึ้นแล้ว (หลัง near-miss)

```
trace_a = [(x, y, frame), ..., (x, y, frame)]   ← เก็บใน direction_memory
trace_b = [(x, y, frame), ..., (x, y, frame)]

วน segment ทุกคู่ของสอง trace:
  ถ้า segment_a[i] ตัดกับ segment_b[j]:
    interpolate frame ที่แต่ละตัวผ่านจุดตัด
    PET = |frame_A - frame_B| / fps

เอา PET ที่น้อยที่สุด (conflict ที่ใกล้ที่สุด)
```

**เหตุที่ต้องเก็บ `frame_count` ใน direction_memory:**  
เดิม `direction_memory` เก็บแค่ `(x, y)` → interpolate เวลาไม่ได้  
เปลี่ยนเป็น `(x, y, frame_count)` เพื่อให้รู้ว่าแต่ละ point ผ่านที่ frame เท่าไร

**Threshold:**

| ตัวแปร | ค่า default | ความหมาย |
|---|---|---|
| `PET_THRESHOLD` | 2.0 s | ถ้า PET < นี้ถือว่า near-miss |

---

### 4. Risk Event Logging

**Cooldown per pair per event type:**

```python
last_risk_logged: dict  # key = (frozenset{id_a, id_b}, event_type)
RISK_COOLDOWN_S = 5.0   # เว้น 5 วิก่อน log คู่เดิมซ้ำ
```

ป้องกัน log ระเบิดเมื่อ condition เป็นจริงหลาย frame ติดกัน  
**เก็บแค่ 1 log ต่อ event** (ไม่ใช่ทุก frame)

**รูปแบบ log:**
```
[2025-07-01 14:23:01] | EVENT_TYPE=TTC_RISK | pair=(#3,#7) | class=(car,motorbike) | TTC=1.83s | zone=Zone_A | frame=412
[2025-07-01 14:23:45] | EVENT_TYPE=PET_RISK | pair=(#3,#9) | class=(car,car)       | PET=0.72s | zone=Zone_A | frame=523
```

**สีบน annotated frame:**
- 🔴 แดง `(0,0,255)` = TTC_RISK — trajectory จะตัดกัน (predictive)
- 🟠 ส้ม `(0,165,255)` = PET_RISK — trace จริงตัดกันแล้ว (retrospective)

---

### 5. Session Config — การถามผู้ใช้ก่อนเริ่ม

เพิ่ม flow การถามผู้ใช้แบบ sequential:

```
Would you like to save a result? [y/N]
  └─ y → สร้าง output folder
         Capture frame as image when risk event occurs? [y/N]
           └─ y → save_frame = True  (บันทึก .jpg ทุก event)
           └─ n → save_frame = False (เก็บแค่ log text)
  └─ n → ไม่บันทึกอะไรเลย
```

`save_frame` จะถูกถามก็ต่อเมื่อ `is_save = True` เท่านั้น

---

### 6. การเปลี่ยนแปลง `direction_memory`

| | เดิม | ใหม่ |
|---|---|---|
| เก็บ | `(x, y)` | `(x, y, frame_count)` |
| maxlen | 20 | 60 |
| เหตุผล | — | PET ต้องการ frame timestamp + trace ยาวขึ้นเพื่อหา intersection |

`estimate_direction()` ถูกแก้ให้ unpack แค่ `(x, y)` จาก tuple ก่อนคำนวณ (รองรับทั้งสองรูปแบบ)

---

### 7. `handle_risk_event()` — signature ใหม่ใน `filemanage.py`

เปลี่ยนจาก:
```python
handle_risk_event(annotated_frame, output_dir, detections, result, tracker_id, points, i, cap, save_log, save_frame)
```

เป็น:
```python
handle_risk_event(
    annotated_frame, output_dir, frame_number,
    tracker_id_a, tracker_id_b,
    class_name_a, class_name_b,
    event_type,       # "TTC_RISK" | "PET_RISK"
    metric_value,     # ค่า TTC หรือ PET (วินาที)
    zone_name,
    save_log, save_frame
)
```

ข้อดี: ไม่ต้องส่ง `detections` object ทั้งก้อนเข้าไป → ไม่ต้องคำนวณซ้ำใน function

---

## สิ่งที่ยังต้องทำ / ข้อจำกัดที่รู้อยู่

- [ ] PET อาจ miss near-miss ที่ path เกือบขนานกัน (segment ไม่ตัดพอดี)
- [x] ~~TTC ยังไม่ handle กรณีวัตถุหยุดนิ่ง (speed ≈ 0) ได้ดีนัก~~ — following/head-on ใช้จริง speed ตรง ๆ, intersection ยังใช้ `eff_speed = max(speed, 0.5)`
- [ ] `compute_pet()` มี O(n²) ต่อ segment pair — ถ้า maxlen เพิ่มอีก ควรพิจารณา spatial index
- [ ] ยังไม่มี unit test สำหรับ `compute_ttc()`, `compute_ttc_following()`, `compute_ttc_head_on()` และ `compute_pet()`

---

### 8. TTC — เพิ่มกรณี Following (rear-end) & Head-on

**ปัญหาเดิม:**
TTC แบบ trajectory intersection จับได้แค่กรณี path ตัดกัน → miss 2 กรณีสำคัญ:
1. **ตามกัน (following)** — คันหลังเร็วกว่าคันหน้า → เสี่ยงชนตูด
2. **สวนกัน (head-on)** — วิ่งสวนเลนเดียวกัน → เสี่ยงหน้าชนหน้า

**วิธีแก้:**

ใช้ **dot product ของ real-world direction vector** แยก scenario:

| dot product | ความหมาย | function |
|---|---|---|
| > `DOT_FOLLOWING_MIN` (0.7) | ทิศทางเดียวกัน | `compute_ttc_following()` |
| < `DOT_HEADON_MAX` (-0.7) | สวนทาง | `compute_ttc_head_on()` |
| อื่น ๆ | ตัดกัน (มุม) | `compute_ttc()` (เดิม) |

**Following (rear-end):**

```
1. แปลง direction → real-world, คำนวณ dot product
2. dot ≥ 0.7 → วิ่งทิศเดียวกัน
3. ใช้ทิศเฉลี่ยเป็นแกน → ฉาย separation vector ลงแกน
   → longitudinal gap (ตามแนวขับ)
   → lateral offset (ขวางเลน)
4. lateral > LATERAL_OFFSET_MAX → คนละเลน → ไม่เสี่ยง
5. closing_speed = speed_behind − speed_ahead
   ≤ 0 → ไม่กำลังตามทัน → ไม่เสี่ยง
6. TTC = gap / closing_speed
```

**Head-on (สวนกัน):**

```
1. dot ≤ -0.7 → สวนทาง
2. ใช้ทิศ A เป็นแกน → ฉาย separation vector
   → longitudinal ต้อง > 0 (B อยู่ข้างหน้า A = มุ่งเข้าหากัน)
   → lateral offset → ถ้ามาก = คนละเลน → ไม่เสี่ยง
3. closing_speed = speed_A + speed_B
4. TTC = longitudinal / closing_speed
```

**Threshold ที่เพิ่ม (ปรับได้ใน config):**

| ตัวแปร | ค่า default | ความหมาย |
|---|---|---|
| `LATERAL_OFFSET_MAX` | 2.0 m | ระยะห่างด้านข้างสูงสุดที่ถือว่า "เลนเดียวกัน" |
| `DOT_FOLLOWING_MIN` | 0.7 | dot product ขั้นต่ำสำหรับ "ทิศทางเดียวกัน" |
| `DOT_HEADON_MAX` | −0.7 | dot product สูงสุดสำหรับ "สวนทาง" |

**การแสดงผลบน frame:**
- Intersection → `TTC 2.1s` (เหมือนเดิม)
- Following → `TTC 2.1s [R]` (R = Rear-end)
- Head-on → `TTC 1.5s [H]` (H = Head-on)

**Log format (ตัวอย่าง):**
```
[2025-07-01 14:23:01] | EVENT_TYPE=TTC_RISK | pair=(#3,#7) | class=(car,car) | TTC=1.83s | type=intersection | zone=Zone_A | frame=412
[2025-07-01 14:23:12] | EVENT_TYPE=TTC_RISK | pair=(#5,#8) | class=(car,motorbike) | TTC=2.41s | type=following | zone=Zone_A | frame=540
[2025-07-01 14:23:30] | EVENT_TYPE=TTC_RISK | pair=(#2,#9) | class=(car,car) | TTC=1.10s | type=head_on | zone=Zone_B | frame=703
```

**ฟังก์ชันใหม่:**
- `compute_ttc_following()` — TTC สำหรับ rear-end (ตามกัน)
- `compute_ttc_head_on()` — TTC สำหรับ head-on (สวนกัน)

**การเปลี่ยนแปลงใน `filemanage.py`:**
- `handle_risk_event()` เพิ่ม parameter `collision_type` (optional) — ใส่ใน log เป็น `type=...`