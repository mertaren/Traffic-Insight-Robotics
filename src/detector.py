from ultralytics import YOLO
import cv2
import numpy as np

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("model loading..")
        self.model = YOLO(model_path)

        # Target classes  (COCO database)
        # 2: car , 3: motorcycle, 5: bus, 7: truck       
        self.target_classes = [2, 3, 5, 7] 

    def detect(self, frame):
        """
        Input: Frame
        Output: Bounding box list
        """
        results = self.model(frame, verbose=False,
                             imgsz=1280, conf=0.30,
                             classes = self.target_classes)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) # our score

            x1, y1, x2 ,y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            detections.append([x1, y1, w, h, confidence, class_id])
        
        return np.array(detections)


