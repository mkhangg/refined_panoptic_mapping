#! usr/bin/python3

import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(4)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("models/yolov8n-seg.pt")
objects_cls = model.model.names

# ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#   "teddy bear", "hair drier", "toothbrush"
#   ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, objects_cls[int(box.cls[0])], [x1, y1], 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('RealSense D435i', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
