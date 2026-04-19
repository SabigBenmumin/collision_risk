# 📄 สรุปเนื้อหา Paper อย่างละเอียด
## Traffic Collision Risk Detection using Computer Vision and Deep Learning

> เอกสารนี้สรุปเนื้อหาจาก `main_full_v2.tex` อย่างละเอียดทุกบท เป็นภาษาไทย  
> เหมาะสำหรับใช้ทบทวนก่อนสอบปากเปล่า / เตรียมนำเสนอ / ตรวจทานเนื้อหา

---

# บทที่ 1: บทนำ (Introduction)

## 1.1 ที่มาและแรงจูงใจ

สถานการณ์ปัจจุบัน:
- องค์การอนามัยโลก (WHO) ประมาณว่ามีผู้เสียชีวิตจากอุบัติเหตุทางถนนประมาณ **1.19 ล้านคนต่อปี** ทั่วโลก
- การเข้าใจ **traffic conflict** (สถานการณ์ที่เกือบจะชนกัน) ช่วยให้วิศวกรจราจรและนักวางผังเมืองสามารถออกแบบถนนที่ปลอดภัยขึ้นได้

วิธีเดิมที่มีปัญหา:
- **คนสังเกตด้วยตา**: เหนื่อย ทำได้จำกัด ไม่ scale
- **ดูวิดีโอย้อนหลัง**: ใช้เวลานาน
- **อุปกรณ์เฉพาะทาง** (radar gun, inductive loop): แพง ต้องติดตั้งบนถนน
- ระบบ video-based ที่มีอยู่ส่วนใหญ่เป็น **black box** → ตรวจสอบไม่ได้ว่าคำนวณยังไง

สิ่งที่โปรเจคนี้ทำ:
- พัฒนา pipeline อัตโนมัติครบวงจร ตั้งแต่อ่านวิดีโอ → ตรวจจับรถ → ติดตาม → วัดความเร็ว → คำนวณ TTC/PET → log ผลลัพธ์
- ใช้แค่กล้อง CCTV ตัวเดียว ไม่ต้องติดตั้ง sensor บนถนน

## 1.2 ปัญหาที่ต้องแก้

1. **แปลง pixel → เมตร** ภายใต้ perspective distortion ของกล้อง
2. **วัดความเร็ว** จาก tracker output ที่มี noise โดยไม่มี ground-truth odometry
3. **ทำนายตำแหน่งล่วงหน้า** ใน metric space แล้วแปลงกลับเป็น pixel
4. **คำนวณ TTC** สำหรับ 3 รูปแบบการชน: ตัดกัน / ตามกัน / สวนกัน
5. **คำนวณ PET** จาก trace path ย้อนหลัง
6. **ป้องกัน numerical instability** ของ homography ตอนวัตถุอยู่ขอบ zone

## 1.3 วัตถุประสงค์การวิจัย

1. พัฒนาระบบ AI ตรวจจับยานพาหนะหลายประเภทแบบ real-time
2. ติดตาม trajectory ของยานพาหนะแต่ละคันข้ามเฟรม
3. วัดความเร็วจริง (km/h) ผ่าน homography + IQR + median
4. ทำนายตำแหน่งล่วงหน้า 1 วินาที
5. คำนวณ **TTC เชิงปริมาณ** 3 รูปแบบ
6. คำนวณ **PET** จาก trace path intersection
7. **จำแนกและบันทึก** risk event พร้อมประเภทการชน

## 1.4 ขอบเขตงาน

**สิ่งที่ทำ:**
- ตรวจจับด้วย YOLOv8 (confidence threshold = 0.65)
- ติดตามด้วย ByteTrack (ผ่าน Supervision library)
- วัดความเร็วเฉพาะใน calibration zone, นอก zone ใช้ค่าล่าสุด
- direction memory 60 frames (เก็บ x, y, frame_count)
- TTC 3 วิธี + PET + risk logging พร้อม collision type

**ข้อจำกัด:**
- สมมติถนนแบน (homography ใช้ได้แค่พื้นราบ)
- ความเร็วอาจผิดพลาดตอนวัตถุอยู่ขอบ zone
- `LATERAL_OFFSET_MAX = 2.0m` อาจต้องปรับตามเลนจริง
- PET มี O(n²) ต่อ segment pair
- ขึ้นอยู่กับ hardware ที่มี

## 1.5 ประโยชน์ที่คาดว่าจะได้รับ

- เข้าใจ near-collision ได้ดีขึ้นด้วยการ monitor อัตโนมัติ 24/7
- ต้นทุนต่ำ — ใช้แค่กล้อง
- ได้ข้อมูลความเร็วเชิงปริมาณ
- TTC/PET metrics ตรงตามมาตรฐาน surrogate safety
- ขยายต่อได้ทั้ง heatmap, cloud-based reporting

---

# บทที่ 2: การทบทวนวรรณกรรม (Literature Review)

## 2.1 วิธีเก็บข้อมูลจราจรแบบดั้งเดิม

| วิธี | ข้อดี | ข้อเสีย |
|---|---|---|
| **Pneumatic road tube** (สายยางบนถนน) | วัดจำนวน + ความเร็วได้ | ต้องติดบนถนน, กีดขวาง |
| **Inductive loop** (ขดลวดฝังถนน) | ทนทาน, ทำงานอัตโนมัติ | แพง, ต้องขุดถนน |
| **Radar speed gun** | แม่นมาก | ต้องมีคนยิงทีละคัน |
| **คนสังเกต** (Swedish TCT) | ข้อมูลคุณภาพสูง | เหนื่อย, ไม่ scale |
| **Background subtraction** (GMM/ViBe) | ไม่ต้องติดตั้งอะไร | ไวต่อแสง/เงา, ไม่แยกรถแต่ละคัน |

## 2.2 Computer Vision กับ Traffic Analysis

- **Background subtraction**: GMM, ViBe → แยก foreground ได้ แต่ไวต่อแสง
- **Optical flow**: Lucas-Kanade, Farneback → วัด motion per-pixel แต่ไม่รู้ว่าเป็นรถคันไหน
- **Homography**: แปลง pixel → เมตร ได้เมื่อพื้นราบ → รากฐานของการวัดความเร็ว

## 2.3 Deep Learning สำหรับ Object Detection

วิวัฒนาการของ detector:

```
R-CNN (2014) → Fast R-CNN → Faster R-CNN → YOLO (2016) → ... → YOLOv8 (2023)
```

- **YOLOv8** (Ultralytics): single-stage, anchor-free, decoupled head
- mAP > 50% บน COCO, real-time inference
- สำหรับ traffic → fine-tune บน dataset เฉพาะทาง (UA-DETRAC, CARLA)

## 2.4 Object Tracking

วิวัฒนาการของ tracker:

```
SORT (2016) → DeepSORT (2017) → ByteTrack (2022)
```

- **SORT**: nearest-neighbor + Kalman filter → ง่ายแต่แตก track ง่าย
- **DeepSORT**: เพิ่ม appearance embedding → ทน occlusion ดีขึ้น
- **ByteTrack**: ใช้ two-stage association → recovery detection ที่ confidence ต่ำ → recall สูงขึ้นมาก โดยไม่ต้องใช้ appearance features → เหมาะกับ traffic ที่รถซ้อนกันบ่อย

## 2.5 Homography

- Homography matrix **H** (3×3): แปลงระหว่าง 2 views ของ planar surface เดียวกัน
- ต้องการอย่างน้อย **4 จุดสมนัย** (point correspondences)
- คำนวณด้วย **DLT** (Direct Linear Transform) → SVD
- ใช้กันมากในงาน traffic monitoring: แปลง pixel displacement → ระยะทางจริง → ความเร็ว

## 2.6 Surrogate Safety Measures

ตัววัดความเสี่ยงทดแทน (ไม่ต้องรอให้ชนจริง):

| ตัววัด | ลักษณะ | คำอธิบาย | อยู่ในโค้ด? |
|---|---|---|---|
| **TTC** (Time to Collision) | Predictive | เวลาที่เหลือก่อนจะชนกัน ถ้ายังคงความเร็ว/ทิศทางเดิม | ✅ |
| **PET** (Post Encroachment Time) | Retrospective | เวลาที่ห่างกันระหว่างที่ 2 วัตถุผ่านจุดเดียวกัน | ✅ |
| **DRAC** (Deceleration Rate to Avoid Crash) | Predictive | อัตราเร่งที่ต้องเบรกเพื่อหลบ | ❌ |

ปัญหาของ TTC แบบเดิม:
- ส่วนใหญ่คำนวณแค่กรณี **เส้นทางตัดกัน** (intersection)
- ไม่ครอบคลุม **following** (ชนตูด) และ **head-on** (สวนกัน)
- ทั้ง 2 กรณีนี้เป็นสัดส่วนสำคัญของอุบัติเหตุจริง

## 2.7 ช่องว่างที่งานนี้เติมเต็ม (Gap Analysis)

งานที่มีอยู่ (Zangenehpour 2015, Romero 2020) ขาด:
1. IQR outlier filter สำหรับ speed
2. Dot-product direction consistency check ทั้ง pixel + world space
3. Pixel sanity check ป้องกัน projection ระเบิด
4. **Multi-scenario TTC** (intersection + following + head-on)
5. **PET จาก trace path intersection**

→ งานนี้ contribute ครบทั้ง 5 ข้อ

---

# บทที่ 3: ระเบียบวิธี (Methodology)

## 3.1 สถาปัตยกรรมระบบ

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Video Input │────▶│   Camera     │────▶│   YOLOv8     │
│              │     │ Calibration  │     │  Detection   │
└──────────────┘     │ (Homography) │     │ (conf=0.65)  │
                     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌──────────────┐     ┌──────▼───────┐
                     │   Speed      │◀────│  ByteTrack   │
                     │ Estimation   │     │  Tracking    │
                     │ (IQR+Median) │     └──────────────┘
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐     ┌──────────────┐
                     │  Direction   │────▶│    Risk      │
                     │ Estimation   │     │  Detection   │
                     │ (Weighted)   │     │ TTC + PET    │
                     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌──────────────┐     ┌──────▼───────┐
                     │  Log File    │◀────│Visualization │
                     │ risk_log.txt │     │ + Annotation │
                     └──────────────┘     └──────────────┘
```

ลำดับการทำงานต่อ 1 frame:
1. **Calibration** (ครั้งเดียว): ผู้ใช้เลือก 4 จุด + ใส่ระยะจริง → ได้ H
2. **Detection**: YOLOv8 หารถใน frame → bounding box + class + confidence
3. **Tracking**: ByteTrack จับคู่ detection กับ tracker ID ข้ามเฟรม
4. **Speed**: แปลง bottom-center → real-world → หาระยะ → หารเวลา
5. **Direction**: weighted displacement → normalize → dot-product check
6. **Risk Detection**: คำนวณ TTC (3 วิธี) + PET ทุกคู่
7. **Visualization + Logging**: วาดบน frame + เขียน log

## 3.2 การเก็บข้อมูลวิดีโอ

- กล้อง fixed overhead ตั้งเหนือถนน
- ความละเอียด 1920×1080
- วัดระยะจริงบนพื้นถนนด้วยตลับเมตรก่อนถ่าย
- FPS จริงอ่านจาก `cap.get(cv2.CAP_PROP_FPS)`

## 3.3 Camera Calibration & Homography

### 3.3.1 Homography Model — ทฤษฎี

ถ้าจุดทั้งหมดอยู่บนระนาบเดียวกัน (Z = 0) กล้อง perspective จะลดเป็น **planar homography**:

```
λ [u, v, 1]ᵀ = H [X, Y, 1]ᵀ
```

- `λ` = projective depth (scalar ≠ 0)
- `H` = homography matrix (3×3), มี 8 degrees of freedom (up to scale)
- ใช้ได้เมื่อ **พื้นถนนแบน** (ข้อสมมติหลัก)

### 3.3.2 ขั้นตอนการ Calibrate

1. ผู้ใช้เลือก **4 จุดบนภาพ** ตามลำดับ: บนซ้าย → บนขวา → ล่างขวา → ล่างซ้าย
2. กำหนด **จุดในพื้นที่จริง**:
   ```
   P₁ = (0, 0)    P₂ = (W, 0)    P₃ = (W, L)    P₄ = (0, L)
   ```
   โดย W = ความกว้าง (เมตร), L = ความยาว (เมตร)
3. มี 3 วิธี input:
   - **Mouse click** บน first frame
   - **อ่านจากไฟล์ .txt** (4 บรรทัด, x,y)
   - **พิมพ์ผ่าน keyboard**

### 3.3.3 Direct Linear Transform (DLT)

สำหรับแต่ละคู่สมนัย (pᵢ, Pᵢ) ได้ 2 สมการเชิงเส้น:

```
[-Xᵢ -Yᵢ -1  0   0   0  uᵢXᵢ uᵢYᵢ uᵢ] h = 0
[ 0   0   0 -Xᵢ -Yᵢ -1  vᵢXᵢ vᵢYᵢ vᵢ] h = 0
```

- 4 จุด → 8 สมการ → A (8×9) matrix
- h = vectorized H (9 ค่า)
- แก้ด้วย **SVD**: h = last column ของ V
- reshape → H (3×3)

### 3.3.4 การแปลงพิกัด

**Pixel → World (forward):**
```python
transformed = cv2.perspectiveTransform(pt, H)
# ได้ (X_world, Y_world) ในหน่วยเมตร
```

**World → Pixel (inverse, สำหรับวาด future point กลับลง frame):**
```python
H_inv = np.linalg.inv(H)
pixel = cv2.perspectiveTransform(world_pt, H_inv)
```

## 3.4 Object Detection

- ใช้ **YOLOv8** กับ model `best.pt` (fine-tuned บน traffic dataset)
- `conf=0.65` เพื่อลด false positive
- ทำงานแบบ **streaming** → ไม่ต้อง buffer ทั้งวิดีโอ

```python
results = model.predict(source=frame, stream=True, conf=0.65)
```

- แปลงผลเป็นรูปแบบ Supervision → ส่งต่อให้ ByteTrack

## 3.5 Object Tracking

- ใช้ **ByteTrack** (Zhang et al., 2022) ผ่าน Supervision library
- ตั้ง `frame_rate` ตาม FPS จริง → Kalman filter calibrate ถูกต้อง
- แต่ละ detection ได้ **tracker_id** ที่คงที่ข้ามเฟรม
- ใช้ **bottom-center anchor** (จุดล่างสุดกลาง bounding box) เป็นจุดสัมผัสพื้น → ลด parallax error

วาด trace ด้วย:
```python
trace_annotator = sv.TraceAnnotator(trace_length=int(FPS*2), ...)
# trace ยาว 2 วินาที
```

## 3.6 Speed Estimation — ละเอียด

### 3.6.1 คำนวณระยะทางจริง

```
d = √((X₂ − X₁)² + (Y₂ − Y₁)²)
```
โดย (Xₖ, Yₖ) = พิกัดจริง (เมตร) จาก homography

### 3.6.2 ความเร็วชั่วขณะ

- ใช้ **sliding deque ขนาด 5** เก็บตำแหน่งล่าสุดใน zone
- คำนวณ:
  ```
  v_inst = d(p_first, p_last) / (n / FPS)   [m/s]
  v_kmh = v_inst × 3.6                       [km/h]
  ```

### 3.6.3 IQR Outlier Filter (ทำตอนรถออกจาก zone)

ปัญหา: ตอนรถอยู่ขอบ zone, homography projection อาจไม่เสถียร → ค่าความเร็วกระโดด

วิธีแก้ — **Tukey fence**:
```
Q₁ = percentile(S, 25)
Q₃ = percentile(S, 75)
IQR = Q₃ − Q₁
ช่วงที่ยอมรับ = [Q₁ − 1.5·IQR, Q₃ + 1.5·IQR]
ค่าที่อยู่นอกช่วง → ตัดทิ้ง
```

### 3.6.4 Median Smoothing

```
v_final = median(S_filtered)
```

**ทำไมใช้ median ไม่ใช้ mean?**
- Median มี **breakdown point = 50%** → ครึ่งหนึ่งของข้อมูลจะผิดพลาดก็ยังไม่กระทบ
- Mean มี breakdown point = 0% → ค่าผิดปกติตัวเดียวก็ดึงค่าเฉลี่ยหลุด
- Homography ใกล้ขอบ zone สร้าง extreme reading ได้ → median ทนได้ดีกว่ามาก

### 3.6.5 Running Median ขณะอยู่ใน zone

แสดงค่าความเร็วขณะที่รถยังอยู่ใน zone:
```
v_display(t) = median(S(1:t))    เมื่อ |S| ≥ 3
```
ถ้ามีข้อมูลน้อยกว่า 3 จุด → แสดงค่า instantaneous

## 3.7 Direction Estimation — ละเอียด

### 3.7.1 Temporally Weighted Displacement

เก็บ track history ไว้ **60 frames** (เป็น tuple `(x, y, frame_count)`)

คำนวณ weighted displacement:
```
ṽ = Σ(k=1→n-1) [k · Δpₖ] / Σ(k=1→n-1) k

โดย Δpₖ = pₖ₊₁ − pₖ
     k = น้ำหนัก (เพิ่มเชิงเส้น ตัวล่าสุดมีน้ำหนักมากสุด)
```

Normalize:
```
v̂ = ṽ / ‖ṽ‖
```

**ทำไมใช้ weighted average ไม่ใช่แค่ดูจุดสุดท้าย?**
- จุดสุดท้ายอย่างเดียว → noise มาก (tracker กระตุก)
- Simple average → ให้น้ำหนักเท่ากันกับทิศทางเก่าๆ → ไม่ตอบสนองเวลาเลี้ยว
- Weighted average → ทั้ง smooth และตอบสนอง

### 3.7.2 Dot-Product Consistency Check

ปัญหา: เมื่อรถวิ่งเป็นโค้ง weighted average อาจให้ vector ชี้กลับทาง

วิธีแก้:
```
δ = p_last − p_first    (overall displacement)
ถ้า v̂ · δ < 0:
    v̂ = −v̂             (กลับทิศ)
```

**ทำ 2 ครั้ง:**
1. ใน **pixel space** (ตอน `estimate_direction()`)
2. ใน **real-world space** (ตอนแปลงผ่าน homography ใน `_get_real_direction_vector()`)

เพราะ projective mapping เป็น non-linear → อาจกลับทิศอีกรอบได้หลังแปลง

### 3.7.3 แปลง Direction Vector → Real-World

```python
pt_base = transform_point((x0, y0), H)
pt_tip  = transform_point((x0 + vx*100, y0 + vy*100), H)
real_v = normalize(pt_tip - pt_base)
```

ยิง pixel point 2 จุดตาม direction แล้วแปลงทั้งคู่ → ได้ direction vector ใน real-world → ใช้กับ TTC ทุกรูปแบบ

### 3.7.4 Future Position Projection

```
Q_future = Q_curr + v · τ · v̂_w
```
- τ = 1 วินาที (prediction horizon)
- แปลงกลับเป็น pixel ด้วย H⁻¹

**Sanity check:**
```
ถ้า ‖p_future − p_current‖ > 500 px → ไม่น่าเชื่อถือ → fallback เป็นเส้นตรง 60 px
```

ป้องกัน projection ระเบิดตอน projective depth (w') เข้าใกล้ 0 ที่ขอบ zone

---

## 3.8 💥 Collision Risk Detection — ส่วนสำคัญที่สุดของ Paper

### 3.8.1 ภาพรวม

ระบบใช้ **2 ตัววัด** เสริมกัน:

| ตัววัด | ลักษณะ | จุดแข็ง |
|---|---|---|
| **TTC** | ทำนายล่วงหน้า (predictive) | เตือนก่อนเกิดเหตุ |
| **PET** | วัดย้อนหลัง (retrospective) | ไม่มี prediction error, ยืนยัน near-miss จริง |

TTC มี **3 รูปแบบ** ตามรูปทรงเรขาคณิตของการชน

### 3.8.2 ขั้นตอนที่ 1: จำแนกประเภทด้วย Dot Product

สำหรับทุกคู่วัตถุ (A, B) คำนวณ:

```
cos θ = v̂_A · v̂_B
```

โดย v̂_A, v̂_B คือ **real-world direction unit vector** ที่ผ่าน homography แล้ว

ตาราง classification:

```
cos θ ≥  0.7  →  วิ่งทิศเดียวกัน  →  compute_ttc_following()
cos θ ≤ -0.7  →  วิ่งสวนทาง       →  compute_ttc_head_on()
อื่นๆ          →  วิ่งตัดกัน (มุม)  →  compute_ttc()
```

**ทำไมใช้ 0.7?**
- cos(45°) ≈ 0.707
- หมายความว่ามุมระหว่าง 2 ทิศทาง ≤ ~45° ถือว่า "ทิศเดียวกัน"
- มุม ≥ ~135° ถือว่า "สวนทาง"
- ค่านี้ปรับได้ใน config

### 3.8.3 ขั้นตอนที่ 2: Lateral Offset Filtering

**ปัญหา**: รถที่วิ่งคู่กันคนละเลน → dot product สูง → อาจถูก flag เป็น "following" ทั้งที่ไม่เสี่ยง

**วิธีแก้**: ฉาย separation vector ลงแกน longitudinal + lateral

```
d̂ = reference direction axis (ทิศเดียวกับการขับ)
d̂⊥ = (-d̂_y, d̂_x)  (ตั้งฉาก)

s = Q_B − Q_A  (separation vector ใน real-world)

d_long = s · d̂         (ระยะตามแนวขับ — ใครอยู่หน้า/หลัง)
d_lat  = |s · d̂⊥|      (ระยะขวางเลน — อยู่เลนเดียวกันมั้ย)
```

- ถ้า `d_lat > 2.0 เมตร` → **คนละเลน** → ไม่ flag เป็นความเสี่ยง
- ค่า 2.0m เหมาะกับเลนทั่วไป ~3.0-3.5m (ครึ่งเลน + offset)
- ปรับได้ตามถนนจริง

### 3.8.4 TTC แบบที่ 1: Trajectory Intersection (เส้นทางตัดกัน)

**เมื่อไรใช้**: รถวิ่งเข้ามาจากคนละทิศ (เช่น สี่แยก, ทางแยก)

**ขั้นตอนละเอียด:**

1. **Project future path** ของแต่ละคัน ออกไป `τ_look = 4.0 วินาที` ใน real-world space:
   ```
   Q_future_A = Q_A + v̂_A · max(v_A, 0.5) · 4.0
   Q_future_B = Q_B + v̂_B · max(v_B, 0.5) · 4.0
   ```
   ใช้ `max(speed, 0.5)` เป็น floor เพื่อไม่ให้ segment มีความยาว 0

2. **Ray-segment intersection test**: ยิง ray จาก A ตาม direction A → ทดสอบว่าตัดกับ segment (B_now → B_future) หรือไม่
   ```
   result = _ray_segment_intersection_2d(
       ray_origin = Q_A, ray_dir = v̂_A,
       seg_start = Q_B, seg_end = Q_B_future
   )
   ```
   ถ้า `result = None` → path ไม่ตัดกัน → **ไม่เสี่ยง** (เช่น วิ่งสวนคนละเลน, ขนาน) → return None

3. **หา conflict point C**:
   ```
   C = Q_A + t_ray · v̂_A
   ```

4. **คำนวณ TTC ของแต่ละคัน**:
   ```
   TTC_A = dist(Q_A, C) / eff_speed_A
   TTC_B = dist(Q_B, C) / eff_speed_B
   ```

5. **เช็คเงื่อนไขเสี่ยง** (ต้องผ่านทั้ง 2):
   ```
   เงื่อนไข 1: TTC_A < 3.0s หรือ TTC_B < 3.0s     (อย่างน้อยคันนึงจะถึงเร็ว)
   เงื่อนไข 2: |TTC_A − TTC_B| < 1.5s               (ถึงพร้อมกันพอสมควร)
   ```

**ตัวอย่าง:**
- รถ A วิ่งจากซ้ายไปขวา 40 km/h
- รถ B วิ่งจากบนลงล่าง 35 km/h
- path ตัดกันที่จุดกลางแยก
- TTC_A = 2.1s, TTC_B = 2.4s
- |2.1 − 2.4| = 0.3 < 1.5 ✓
- min(2.1, 2.4) = 2.1 < 3.0 ✓
- → **TTC_RISK**, collision_type = "intersection"

### 3.8.5 TTC แบบที่ 2: Following / Rear-End (ชนตูด)

**เมื่อไรใช้**: dot product ≥ 0.7 (วิ่งทิศเดียวกัน)

**ขั้นตอนละเอียด:**

1. **คำนวณทิศเฉลี่ย** เป็นแกนอ้างอิง:
   ```
   d̂_avg = normalize((v̂_A + v̂_B) / 2)
   ```
   ใช้ทิศเฉลี่ยเพราะทั้งคู่ไม่จำเป็นต้องชี้ไปทิศเดียวกัน 100% (อาจห่างกันเล็กน้อย)

2. **ฉาย separation vector**:
   ```
   s = Q_B − Q_A
   d_long = s · d̂_avg           (ใครอยู่หน้า/หลัง)
   d_lat  = |s · d̂_avg⊥|        (ระยะขวางเลน)
   ```

3. **เช็ค lateral**: ถ้า > 2.0m → คนละเลน → ข้าม

4. **เช็ค gap**: ถ้า |d_long| < 0.3m → ใกล้เกินไป ข้อมูลไม่น่าเชื่อถือ → ข้าม

5. **หาใครอยู่หลัง + คำนวณ closing speed**:
   ```
   ถ้า d_long > 0:    A อยู่หลัง B → closing = speed_A − speed_B
   ถ้า d_long < 0:    B อยู่หลัง A → closing = speed_B − speed_A
   ```
   ถ้า closing ≤ 0.1 m/s → **ไม่กำลังตามทัน** → ข้าม

6. **คำนวณ TTC**:
   ```
   TTC = |d_long| / closing_speed
   ```

**ตัวอย่าง:**
- รถ A (คันหลัง) วิ่ง 60 km/h = 16.67 m/s
- รถ B (คันหน้า) วิ่ง 40 km/h = 11.11 m/s
- gap = 15 เมตร
- lateral = 0.5m (เลนเดียวกัน)
- closing = 16.67 − 11.11 = 5.56 m/s
- TTC = 15 / 5.56 = **2.7 วินาที** < 3.0 → เสี่ยง!
- → **TTC_RISK**, collision_type = "following", แสดง `TTC 2.7s [R]`

### 3.8.6 TTC แบบที่ 3: Head-On (สวนกัน)

**เมื่อไรใช้**: dot product ≤ −0.7 (วิ่งสวนทาง)

**ขั้นตอนละเอียด:**

1. **ใช้ทิศของ A เป็นแกนอ้างอิง**: (ไม่ใช้ค่าเฉลี่ยเพราะสวนกัน → เฉลี่ยแล้วจะเกือบ 0)
   ```
   ref_d = v̂_A
   ```

2. **ฉาย separation**:
   ```
   d_long = s · v̂_A
   ```
   - ต้อง > 0 → หมายความว่า B อยู่ **ข้างหน้า** A ในทิศที่ A กำลังมุ่งไป = ยังมุ่งเข้าหากัน
   - ถ้า ≤ 0.3 → ผ่านกันไปแล้ว หรือใกล้เกินไป → ข้าม

3. **เช็ค lateral**: เหมือนเดิม > 2.0m → คนละเลน → ข้าม

4. **Closing speed = ผลรวม** (เพราะมุ่งเข้าหากัน):
   ```
   closing = speed_A + speed_B
   ```

5. **คำนวณ TTC**:
   ```
   TTC = d_long / closing
   ```

**ตัวอย่าง:**
- รถ A วิ่ง 50 km/h = 13.89 m/s ไปทางขวา
- รถ B วิ่ง 45 km/h = 12.50 m/s ไปทางซ้าย (สวนกัน)
- longitudinal gap = 30 เมตร
- lateral = 0.8m (อยู่เลนเดียวกัน → อันตรายมาก!)
- closing = 13.89 + 12.50 = 26.39 m/s
- TTC = 30 / 26.39 = **1.14 วินาที** → อันตรายมาก!
- → **TTC_RISK**, collision_type = "head_on", แสดง `TTC 1.1s [H]`

### 3.8.7 ลำดับการประเมิน (Priority Order)

สำหรับแต่ละคู่ในแต่ละ frame:

```python
# 1) ลอง Intersection ก่อน
ttc_result = compute_ttc(...)
if ttc_result: collision_type = "intersection"

# 2) ถ้าไม่ใช่ → ลอง Following
if not ttc_result:
    ttc_result = compute_ttc_following(...)
    if ttc_result: collision_type = "following"

# 3) ถ้ายังไม่ใช่ → ลอง Head-on
if not ttc_result:
    ttc_result = compute_ttc_head_on(...)
    if ttc_result: collision_type = "head_on"
```

**ทำไมลอง intersection ก่อน?**
- เป็น case ที่เข้มงวดที่สุด (path ต้องตัดกันจริงๆ)
- ถ้า path ตัดกัน → แสดงว่ามีจุด conflict ชัดเจน → ควรรายงานก่อน
- Following/head-on เป็น fallback สำหรับกรณีที่ path ไม่ตัดกันแต่ยังเสี่ยง

### 3.8.8 PET: Post Encroachment Time

**แตกต่างจาก TTC อย่างไร?**

```
TTC: "ถ้ายังคงเร็วและทิศทางเดิม จะชนกันอีกกี่วินาที?" → ทำนาย
PET: "ทั้งคู่ผ่านจุดเดียวกัน ห่างกันกี่วินาที?"          → วัดจริง
```

**ขั้นตอน:**

1. ทุกวัตถุเก็บ trace เป็น `(x, y, frame_count)` สูงสุด **60 จุด**

2. สำหรับทุกคู่ segment ใน trace A × trace B:
   ```
   segment A: (A_i → A_{i+1})
   segment B: (B_j → B_{j+1})
   ```

3. ทดสอบว่า 2 segment ตัดกันไหม ด้วย `_segment_intersection()`:
   ```
   result = (t, u)   โดย 0 ≤ t,u ≤ 1
   ```

4. ถ้าตัด → interpolate frame ที่แต่ละตัวผ่านจุดนั้น:
   ```
   f_A = f_{A_i} + t · (f_{A_{i+1}} − f_{A_i})
   f_B = f_{B_j} + u · (f_{B_{j+1}} − f_{B_j})
   ```

5. คำนวณ PET:
   ```
   PET = |f_A − f_B| / FPS
   ```

6. เก็บค่า PET ที่ต่ำสุด (conflict point ที่ใกล้ที่สุด)

7. ถ้า PET < 2.0 วินาที → **PET_RISK**

**ตัวอย่าง:**
- trace A ผ่านจุด (150, 200) ที่ frame 412
- trace B ผ่านจุดเดียวกันที่ frame 440
- PET = |412 − 440| / 30 FPS = 28/30 = **0.93 วินาที** → near-miss!

### 3.8.9 Risk Event Cooldown

ปัญหา: TTC/PET จะถูกคำนวณ**ทุก frame** → ถ้าเสี่ยงต่อเนื่อง 2 วินาที ที่ 30 FPS = 60 log entries!

วิธีแก้:
```
สำหรับแต่ละ (pair_key, event_type):
    ถ้า t_current − t_last_logged ≥ 5.0 วินาที:
        log เหตุการณ์
        อัปเดต t_last_logged = t_current
    มิฉะนั้น:
        ข้าม (ยังอยู่ใน cooldown)
```

## 3.9 Data Logging

ข้อมูลที่บันทึกต่อ 1 event:

| ฟิลด์ | ตัวอย่าง | คำอธิบาย |
|---|---|---|
| Timestamp | 2025-07-01 14:23:01 | เวลาจริง (wall-clock) |
| EVENT_TYPE | TTC_RISK | ประเภท event |
| pair | (#3, #7) | tracker ID ของคู่ |
| class | (car, car) | ประเภทยานพาหนะ |
| TTC/PET | 1.83s | ค่าที่คำนวณได้ |
| type | intersection | ประเภทการชน (TTC เท่านั้น) |
| zone | Zone_A | zone ที่เกิดเหตุ |
| frame | 412 | หมายเลข frame |

ตัวอย่าง log:
```
[2025-07-01 14:23:01] | EVENT_TYPE=TTC_RISK | pair=(#3,#7) | class=(car,car) | TTC=1.83s | type=intersection | zone=Zone_A | frame=412
[2025-07-01 14:23:12] | EVENT_TYPE=TTC_RISK | pair=(#5,#8) | class=(car,motorbike) | TTC=2.41s | type=following | zone=Zone_A | frame=540
[2025-07-01 14:23:30] | EVENT_TYPE=PET_RISK | pair=(#2,#9) | class=(car,car) | PET=1.10s | zone=Zone_B | frame=703
```

---

# บทที่ 4: Implementation & Results

## 4.1 Config Parameters ทั้งหมด

### Risk Detection Parameters

| ตัวแปร | ค่า | หน้าที่ |
|---|---|---|
| `TTC_THRESHOLD` | 3.0 s | ค่า TTC สูงสุดที่ถือว่าเสี่ยง |
| `ARRIVAL_GAP` | 1.5 s | |TTC_A − TTC_B| สูงสุดสำหรับ intersection |
| `PET_THRESHOLD` | 2.0 s | ค่า PET สูงสุดที่ถือว่า near-miss |
| `RISK_COOLDOWN_S` | 5.0 s | หน่วงก่อน log คู่เดิมซ้ำ |
| `TTC_LOOKAHEAD_S` | 4.0 s | ความยาว future path (intersection) |
| `LATERAL_OFFSET_MAX` | 2.0 m | ระยะขวางเลนสูงสุด |
| `DOT_FOLLOWING_MIN` | 0.7 | threshold "ทิศเดียวกัน" |
| `DOT_HEADON_MAX` | −0.7 | threshold "สวนทาง" |

### General Parameters

| ตัวแปร | ค่า | หน้าที่ |
|---|---|---|
| Detection confidence | 0.65 | ตัด detection ที่ confidence ต่ำ |
| Speed memory | 5 frames | sliding window สำหรับ instantaneous speed |
| Direction memory | 60 frames | trace history สำหรับ direction + PET |
| Prediction horizon | 1 second | ทำนายล่วงหน้ากี่วินาที |
| Sanity distance | 500 px | future point ไกลกว่านี้ → fallback |
| Fallback length | 60 px | ความยาวเส้นเมื่อ fallback |
| Edge margin | 50 px | ไม่วาด direction line ตอนอยู่ขอบ frame |

## 4.2 การแสดงผล

| สัญลักษณ์บน Frame | ความหมาย |
|---|---|
| Polygon สีฟ้า | Calibration zone |
| `Speed: XX.X km/h` สีเขียว | ความเร็ว (อยู่ใน zone) |
| `Speed: XX.X km/h` สีส้ม | ความเร็วล่าสุด (ออกจาก zone แล้ว) |
| เส้นสีแดงจากรถ | ทิศทาง + ตำแหน่งล่วงหน้า 1 วินาที |
| เส้นสีแดงเชื่อม 2 รถ + `TTC X.Xs` | TTC risk (intersection) |
| `TTC X.Xs [R]` | TTC risk แบบ rear-end |
| `TTC X.Xs [H]` | TTC risk แบบ head-on |
| เส้นสีส้มเชื่อม 2 รถ + `PET X.Xs` | PET risk (near-miss) |
| มุมซ้ายบน `Frame: X Time: MM:SS.ss` | frame number + เวลา |

---

# บทที่ 5: อภิปราย (Discussion)

## จุดแข็ง 8 ข้อ

1. **ไม่ต้องติดตั้ง sensor บนถนน** — ใช้แค่กล้อง CCTV ที่มีอยู่แล้ว
2. **โปร่งใส (Mathematical Transparency)** — ทุกค่าตรวจสอบย้อนกลับได้: H matrix → calibration inputs → frame timestamps
3. **IQR + Median** — สถิติที่ design มาเพื่อทนต่อ noise โดยเฉพาะ
4. **Multi-scenario TTC** — ครอบคลุม 3 รูปแบบ (ไม่ใช่แค่ intersection อย่างระบบอื่น)
5. **Dual metrics** — TTC (ทำนาย) + PET (ยืนยัน) → ภาพรวมสมบูรณ์กว่าใช้ตัวเดียว
6. **Collision type classification** — log ระบุประเภท → วิเคราะห์สถิติ pattern ได้
7. **Stable visualization** — sanity check + zone restriction → ไม่มี projection ระเบิด
8. **Real-time** — ทำงานได้ที่ frame rate บน consumer GPU

## ข้อจำกัด 7 ข้อ

1. **ถนนต้องแบน** — ทางลาด/สะพาน → homography ผิด → speed ผิด
2. **กล้องตัวเดียว** — ไม่มี depth information → รถสูง/ถูกบังอาจ anchor ไม่อยู่บนพื้น
3. **Manual calibration** — ต้องให้คนเลือกจุด + ใส่ระยะ → ยังไม่ auto
4. **Lateral offset คงที่ 2.0m** — ถนนเลนแคบ/กว้างต่างกัน อาจต้อง tune
5. **PET = O(n²)** — 60 frames OK แต่ถ้าเพิ่ม → ควรมี spatial index
6. **ByteTrack แตก track** — occlusion หนักๆ → ID เปลี่ยน → speed history เสีย
7. **สมมติ constant velocity** — เบรก/เลี้ยวกะทันหัน → TTC ไม่แม่น

## การประยุกต์ใช้

1. **Safety audit ที่ทางแยก** — ติดกล้อง monitor หลายวัน → สถิติ TTC/PET
2. **โซนโรงเรียน** — ตรวจจับรถที่เร็วเกินกำหนดอัตโนมัติ
3. **ปรับสัญญาณไฟ** — ใช้ข้อมูลความเร็ว + ความหนาแน่นแบบ real-time
4. **ฐานข้อมูล near-miss** — log เหตุการณ์พร้อม frame → หลักฐานย้อนหลัง
5. **Before-after study** — เปรียบเทียบ TTC/PET ก่อน/หลังปรับปรุงถนน

---

# บทที่ 6: สรุป (Conclusion)

ระบบนี้รวม 4 เทคโนโลยีหลักเข้าด้วยกัน:

```
YOLOv8 + ByteTrack + Homography + Surrogate Safety Measures (TTC + PET)
```

ผลลัพธ์สำคัญ:
- **H matrix** จาก DLT → แปลง pixel ↔ เมตร
- **Speed** = displacement / time พร้อม IQR + median → ทน noise
- **Direction** = weighted displacement + dot-product check (×2) → stable
- **TTC 3 รูปแบบ** ด้วย dot-product classification + lateral offset filtering
- **PET** จาก trace path intersection + frame interpolation
- **Risk logging** พร้อม collision_type + cooldown

ระบบนี้เป็น **ทางเลือกต้นทุนต่ำ** แทน sensor infrastructure และให้ **ตัววัดเชิงปริมาณ** ที่ใช้ประเมินความปลอดภัยแบบ evidence-based ได้

**งานในอนาคต:**
- Auto calibration จาก lane marking
- TTC ที่คำนึงถึงการเร่ง/เบรก (acceleration-aware)
- PET + spatial index สำหรับ trace ยาว
- Cloud-based incident reporting

---

# Appendix: สิ่งที่ยังต้องเติมข้อมูลจริง

| Section | สิ่งที่ต้องเติม |
|---|---|
| Table 4.2 | Hardware specs: CPU, GPU, RAM |
| Table 4.3 | Processing time จริง per stage |
| Table 4.5 | Speed MAE, RMSE ต่อประเภทรถ |
| Table 4.6 | Detection Precision, Recall, F1, mAP |
| Section 4.3 | จำนวนวิดีโอ, เวลารวม, สถานที่ |
| Section 4.4.2-4.4.5 | Case study ค่าจริง (TTC, dot product, lateral offset) |
| Figure B.1-B.6 | Screenshot จริงจากระบบ (แทน placeholder box) |
