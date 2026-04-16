import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO(r"models/best.pt")
image = cv2.imread(r"img/traffict.png")
CONF_THRESHOLD = 0.7
results = model.predict(source=image, conf=CONF_THRESHOLD)
boxes = results[0].boxes

cv2.imshow("predicted image", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()