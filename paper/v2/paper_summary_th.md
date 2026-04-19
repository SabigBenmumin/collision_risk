# สรุปเนื้อหา Paper: Traffic Collision Risk Detection using Computer Vision and Deep Learning

> อ้างอิงจาก `main_full_v2.tex`

---

## บทที่ 1: บทนำ

### ปัญหา
- อุบัติเหตุทางถนนคร่าชีวิต ~1.19 ล้านคน/ปี (WHO 2023)
- การศึกษา traffic conflict แบบเดิมใช้คนสังเกต → เหนื่อย, แพง, ไม่ scale
- ระบบ video-based ที่มีอยู่มักเป็น black box ไม่โปร่งใสเรื่องวิธีคำนวณ

### เป้าหมาย
1. ตรวจจับยานพาหนะหลายประเภท real-time ด้วย YOLOv8
2. ติดตาม trajectory ด้วย ByteTrack
3. วัดความเร็วจริง (km/h) ผ่าน homography + IQR filter + median
4. ทำนายตำแหน่งล่วงหน้า 1 วินาที
5. คำนวณ **TTC** เชิงปริมาณ 3 รูปแบบ: ตัดกัน / ตามกัน / สวนกัน
6. คำนวณ **PET** จาก trace path ย้อนหลัง
7. บันทึก risk event พร้อมประเภทการชน

---

## บทที่ 3: Methodology (หัวใจหลัก)

### 3.1 Pipeline ทั้งระบบ

```
Video → Camera Calibration → YOLOv8 Detection → ByteTrack Tracking
     → Speed Estimation → Direction Estimation → Risk Detection (TTC + PET)
     → Visualization + Logging
```

### 3.2 Homography (แปลง pixel → เมตร)

- ผู้ใช้เลือก 4 จุดบนถนนที่รู้ระยะจริง
- คำนวณ Homography Matrix **H** (3×3) ด้วย DLT (Direct Linear Transform)
- ใช้ `cv2.perspectiveTransform` แปลงตำแหน่ง pixel → พิกัดจริง (เมตร)
- แปลงกลับด้วย H⁻¹ สำหรับวาด future position กลับลง frame

### 3.3 Speed Estimation

```
ความเร็ว = ระยะจริงระหว่าง 2 จุด (เมตร) / เวลาที่ผ่านไป (วินาที)
```

- ใช้ sliding window 5 frames สำหรับคำนวณความเร็วชั่วขณะ
- **IQR filter**: ตัด outlier ด้วย Tukey fence (Q1 − 1.5·IQR ถึง Q3 + 1.5·IQR)
- **Median smoothing**: ใช้ค่ามัธยฐานแทนค่าเฉลี่ย → ทน noise ดีกว่ามาก

### 3.4 Direction Estimation

- ใช้ weighted average ของ displacement (น้ำหนักเพิ่มตาม recency)
- **Dot-product check**: ป้องกัน direction กลับทิศ (ทำ 2 ครั้ง: pixel space + real-world space)
- แปลง direction จาก pixel → real-world ด้วย homography เพื่อใช้กับ TTC

### 3.5 Pixel Sanity Check

- ถ้า future point ไกลเกิน 500 px → fallback เป็นเส้นตรง 60 px ใน pixel space
- ป้องกัน projection ระเบิดตอนวัตถุอยู่ขอบ zone

---

## 💥 3.6 Collision Risk Detection (ส่วนสำคัญที่สุด)

### ขั้นตอนที่ 1: จำแนกประเภทด้วย Dot Product

คำนวณ dot product ของ direction vector ทั้งคู่ใน real-world space:

```
cos θ = v̂_A · v̂_B
```

| cos θ | หมายความว่า | ฟังก์ชันที่ใช้ |
|---|---|---|
| ≥ 0.7 | วิ่งทิศเดียวกัน (ตามกัน) | `compute_ttc_following()` |
| ≤ −0.7 | วิ่งสวนทาง | `compute_ttc_head_on()` |
| อื่นๆ | เข้ามุม (ตัดกัน) | `compute_ttc()` |

### ขั้นตอนที่ 2: Lateral Offset Filtering (กรองคนละเลน)

```
d_long = separation · d̂        (ระยะตามแนวขับ)
d_lat  = |separation · d̂⊥|     (ระยะขวางเลน)
```

- ถ้า `d_lat > 2.0 เมตร` → ถือว่าคนละเลน → **ไม่ flag เป็นความเสี่ยง**
- สำคัญมากสำหรับป้องกัน false positive ในถนนหลายเลน

---

### TTC แบบที่ 1: Trajectory Intersection (เส้นทางตัดกัน)

**ใช้เมื่อ**: รถวิ่งเข้ามาจากคนละทิศ (เช่น สี่แยก)

**วิธีการ**:
1. ยิง future path ของแต่ละคันออกไป 4 วินาที ใน real-world space
2. ทดสอบว่า ray A ตัดกับ segment B หรือไม่ (ray-segment intersection)
3. ถ้าตัด → ได้ conflict point C
4. คำนวณ:
   ```
   TTC_A = dist(A → C) / speed_A
   TTC_B = dist(B → C) / speed_B
   ```
5. **เสี่ยง** เมื่อ:
   - `TTC_A < 3.0s` หรือ `TTC_B < 3.0s` (ถึงเร็ว)
   - `|TTC_A − TTC_B| < 1.5s` (ถึงพร้อมกัน)

---

### TTC แบบที่ 2: Following / Rear-end (ชนตูด)

**ใช้เมื่อ**: dot product ≥ 0.7 (วิ่งทิศเดียวกัน)

**วิธีการ**:
1. ใช้ทิศเฉลี่ยของทั้งคู่เป็นแกนอ้างอิง
2. ฉาย separation vector ลงแกน → ได้ longitudinal gap + lateral offset
3. เช็ค lateral → ถ้า > 2.0m → คนละเลน → ข้าม
4. หาว่าใครอยู่หลัง:
   ```
   ถ้า A อยู่หลัง B → closing_speed = speed_A − speed_B
   ถ้า B อยู่หลัง A → closing_speed = speed_B − speed_A
   ```
5. ถ้า closing_speed ≤ 0.1 → ไม่กำลังตามทัน → ข้าม
6. `TTC = gap / closing_speed`

---

### TTC แบบที่ 3: Head-on (สวนกัน)

**ใช้เมื่อ**: dot product ≤ −0.7 (วิ่งสวนทาง)

**วิธีการ**:
1. ใช้ทิศของ A เป็นแกนอ้างอิง
2. ฉาย separation → longitudinal ต้อง > 0 (B อยู่ข้างหน้า = ยังมุ่งเข้าหากัน)
3. เช็ค lateral → ถ้า > 2.0m → คนละเลน → ข้าม
4. `closing_speed = speed_A + speed_B` (มุ่งเข้าหากัน)
5. `TTC = longitudinal / closing_speed`

---

### PET: Post Encroachment Time (วัดย้อนหลัง)

**แตกต่างจาก TTC**: TTC ทำนายล่วงหน้า / PET วัดจากสิ่งที่เกิดขึ้นจริงแล้ว

**วิธีการ**:
1. เก็บ trace path เป็น `(x, y, frame_count)` สูงสุด 60 จุดต่อวัตถุ
2. หาจุดตัดของ segment ใน trace A กับ trace B (ทุกคู่)
3. ถ้าตัดกัน → interpolate หา frame ที่แต่ละตัวผ่านจุดนั้น
4. `PET = |frame_A − frame_B| / FPS`
5. เก็บ PET ที่ต่ำสุด → ถ้า < 2.0 วินาที → flag เป็น near-miss

---

### Cooldown (ป้องกัน log ระเบิด)

- แต่ละคู่ (id_A, id_B) + event_type → log ได้ไม่เกิน 1 ครั้งต่อ 5 วินาที
- ป้องกันการ log ซ้ำเหตุการณ์เดิมทุก frame

---

## Config Parameters ที่สำคัญ

| ตัวแปร | ค่า | หน้าที่ |
|---|---|---|
| `TTC_THRESHOLD` | 3.0 s | ค่า TTC สูงสุดที่ถือว่าเสี่ยง |
| `ARRIVAL_GAP` | 1.5 s | ช่วงเวลาถึง conflict point ที่ยอมรับ (intersection) |
| `PET_THRESHOLD` | 2.0 s | ค่า PET สูงสุดที่ถือว่า near-miss |
| `RISK_COOLDOWN_S` | 5.0 s | cooldown ก่อน log คู่เดิมซ้ำ |
| `TTC_LOOKAHEAD_S` | 4.0 s | ความยาว future path (intersection) |
| `LATERAL_OFFSET_MAX` | 2.0 m | ระยะขวางเลนสูงสุดที่ถือว่าเลนเดียวกัน |
| `DOT_FOLLOWING_MIN` | 0.7 | threshold ทิศเดียวกัน |
| `DOT_HEADON_MAX` | −0.7 | threshold สวนทาง |

---

## การแสดงผลบน Frame

| สัญลักษณ์ | ความหมาย |
|---|---|
| **เส้นสีแดง** เชื่อม 2 วัตถุ + `TTC X.Xs` | TTC risk (intersection) |
| `TTC X.Xs [R]` | TTC risk แบบ rear-end (ชนตูด) |
| `TTC X.Xs [H]` | TTC risk แบบ head-on (สวนกัน) |
| **เส้นสีส้ม** เชื่อม 2 วัตถุ + `PET X.Xs` | PET risk (near-miss) |
| **เส้นสีแดง** จากวัตถุ | ทิศทาง + ตำแหน่งล่วงหน้า 1 วินาที |
| **ป้ายสีเขียว** `Speed: XX.X km/h` | ความเร็วขณะอยู่ใน zone |
| **ป้ายสีส้ม** `Speed: XX.X km/h` | ความเร็วล่าสุด (ออกจาก zone แล้ว) |

---

## Log Format ตัวอย่าง

```
[2025-07-01 14:23:01] | EVENT_TYPE=TTC_RISK | pair=(#3,#7) | class=(car,car) | TTC=1.83s | type=intersection | zone=Zone_A | frame=412
[2025-07-01 14:23:12] | EVENT_TYPE=TTC_RISK | pair=(#5,#8) | class=(car,motorbike) | TTC=2.41s | type=following | zone=Zone_A | frame=540
[2025-07-01 14:23:30] | EVENT_TYPE=PET_RISK | pair=(#2,#9) | class=(car,car) | PET=1.10s | zone=Zone_B | frame=703
```

---

## บทที่ 5: จุดแข็ง

1. **ไม่ต้องติดตั้งอะไรบนถนน** — ใช้แค่กล้อง
2. **โปร่งใส** — ทุกค่าตรวจสอบย้อนกลับได้ถึง homography + frame
3. **TTC ครอบคลุม 3 รูปแบบ** — ไม่ใช่แค่เส้นทางตัดกันอย่างเดียว
4. **Dual metrics** — TTC (ทำนาย) + PET (ยืนยันย้อนหลัง) เสริมกัน
5. **Collision type classification** — log ระบุประเภทการชนได้
6. **IQR + Median** — ความเร็วทนต่อ noise ดี

## บทที่ 5: ข้อจำกัด

1. **สมมติถนนแบน** — ถ้ามีทางลาด/สะพานจะมี error
2. **Lateral offset = 2.0m คงที่** — อาจต้อง tune ตามความกว้างเลนจริง
3. **PET = O(n²)** — ถ้า trace ยาวมากจะช้า
4. **ByteTrack อาจแตก track** เมื่อถูกบังมาก
5. **สมมติ constant velocity** — ถ้าเบรก/เลี้ยวกะทันหัน TTC จะไม่แม่น

---

## สิ่งที่ต้องเติมข้อมูลจริง (Placeholder ใน Paper)

- ตาราง Speed MAE, RMSE ต่อประเภทรถ
- ตาราง Detection Precision/Recall/F1/mAP
- Hardware specs (CPU, GPU, RAM)
- Case study ค่าจริง (TTC ที่วัดได้, dot product ที่วัดได้)
- จำนวนวิดีโอ + เวลารวมที่ทดสอบ
- รูป screenshot จริงจากระบบ (แทน placeholder box)
