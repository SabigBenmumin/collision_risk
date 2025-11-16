import cv2
from ultralytics import YOLO
from filemanage import select_model_file
import matplotlib.pyplot as plt

# model_path = select_model_file()
# model = YOLO(model_path)
CONF_THRESHOLD = 0.7
MARGIN_RATIO = 0.15  # เผื่อพื้นที่รอบกรอบ 10%

model = YOLO(r'models/best.pt')
img_source = cv2.imread(r'img/traffict.png')
# img_source = cv2.imread(r'img/first_frame1.png')

# print(img_source.shape)
# cv2.imshow("original image",img_source)
# cv2.waitKey(0)

result = model.predict(source=img_source, conf=CONF_THRESHOLD)
print(result[0].plot().shape)
cv2.imshow("predicted", result[0].plot())
cv2.waitKey(0)

frame = result[0]
boxes = frame.boxes.xyxy.cpu().numpy()
confs = frame.boxes.conf.cpu().numpy()
classes = frame.boxes.cls.cpu().numpy()

h, w, channel = img_source.shape

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
        # ✅ คำนวณ margin รอบกรอบ
    box_w = x2 - x1
    box_h = y2 - y1
    dx = int(box_w * MARGIN_RATIO)
    dy = int(box_h * MARGIN_RATIO)

    # ✅ ขยายกรอบ แต่ไม่ให้เกินขนาดภาพ
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    cropped = img_source[y1:y2, x1:x2]
    cv2.imshow(f"{frame.names[classes[i]]}conf{confs[i]}", cropped)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print(frame.names)