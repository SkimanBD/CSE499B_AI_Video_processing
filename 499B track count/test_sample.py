import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = (x, y)
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('vid2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

# Define the detection point
cy_detection = 424
offset = 6

# Create trackers and counters for all classes
tracker_motorcycle = Tracker()
tracker_car = Tracker()
tracker_truck = Tracker()
tracker_bus = Tracker()

counter_motorcycle = 0
counter_car = 0
counter_truck = 0
counter_bus = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        # Track objects
        if 'motorcycle' in c:
            bbox_idx = tracker_motorcycle.update([[x1, y1, x2, y2]])
            for bbox in bbox_idx:
                x3, y3, x4, y4, id1 = bbox
                cxm = int(x3 + x4) // 2
                cym = int(y3 + y4) // 2
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
                cvzone.putTextRect(frame, 'motorcycle', (x3, y3), 1, 1)

                # Check if the motorcycle crosses the detection zone
                if y3 < cy_detection + offset and y4 > cy_detection - offset:
                    counter_motorcycle += 1

        elif 'car' in c:
            bbox_idx = tracker_car.update([[x1, y1, x2, y2]])
            for bbox in bbox_idx:
                x3, y3, x4, y4, id1 = bbox
                cxm = int(x3 + x4) // 2
                cym = int(y3 + y4) // 2
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
                cvzone.putTextRect(frame, 'car', (x3, y3), 1, 1)

                # Check if the car crosses the detection zone
                if y3 < cy_detection + offset and y4 > cy_detection - offset:
                    counter_car += 1

        elif 'truck' in c:
            bbox_idx = tracker_truck.update([[x1, y1, x2, y2]])
            for bbox in bbox_idx:
                x3, y3, x4, y4, id1 = bbox
                cxm = int(x3 + x4) // 2
                cym = int(y3 + y4) // 2
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
                cvzone.putTextRect(frame, 'truck', (x3, y3), 1, 1)

                # Check if the truck crosses the detection zone
                if y3 < cy_detection + offset and y4 > cy_detection - offset:
                    counter_truck += 1

        elif 'bus' in c:
            bbox_idx = tracker_bus.update([[x1, y1, x2, y2]])
            for bbox in bbox_idx:
                x3, y3, x4, y4, id1 = bbox
                cxm = int(x3 + x4) // 2
                cym = int(y3 + y4) // 2
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
                cvzone.putTextRect(frame, 'bus', (x3, y3), 1, 1)

                # Check if the bus crosses the detection zone
                if y3 < cy_detection + offset and y4 > cy_detection - offset:
                    counter_bus += 1

    cv2.line(frame, (2, cy_detection), (794, cy_detection), (0, 0, 255), 2)

    cvzone.putTextRect(frame, f'motorcycle count: {counter_motorcycle}', (19, 30), 2, 1)
    cvzone.putTextRect(frame, f'car count: {counter_car}', (19, 60), 2, 1)
    cvzone.putTextRect(frame, f'truck count: {counter_truck}', (19, 90), 2, 1)
    cvzone.putTextRect(frame, f'bus count: {counter_bus}', (19, 120), 2, 1)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
