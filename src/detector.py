from ultralytics import YOLO
import cv2
import numpy as np

class VechileDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("model loading..")
        self.model = YOLO(model_path)

        # Taget classes  (COCO database)
        # 2: car , 3: motorcycle, 5: bus, 7: truck       
        self.target_classes = [2, 3, 5, 7] 

    def detect(self, frame):
        """
        Docstring for detect
        Input: Frame
        Output: Bounding box list
        """
        pass

