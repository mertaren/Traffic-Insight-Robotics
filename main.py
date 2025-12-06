import cv2
from src.detector import VehicleDetector
from src.tracker import  DistanceTracker


VIDEO_PATH = "data/videos/testvideo.mp4"

def main():
    detector = VehicleDetector()
    tracker = DistanceTracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Video didn't open..")
        return
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            # inf loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections = detector.detect(frame)
        
        boxes_for_tracker = [] # [[x, y, w , h, ID]]

        for det in detections:
            x, y, w, h, conf, class_id = det
            
            area = w * h # area of the box
            min_area_th = 1500

            if area < min_area_th and class_id != 3:
                continue
            
            # ROI filter for false tracking
            if (630 < x < 840) and (200 < y < 420):
                continue

            boxes_for_tracker.append([x, y, w ,h])

        # call tracker function
        boxes_ids = tracker.update(boxes_for_tracker)
            
        for box_id in boxes_ids:
            x, y, w, h, id = box_id

            cv2.rectangle(frame, (int(x), int(y)),
                        (int(x + w), int(y + h)),
                        (0, 255 ,0), 2)
            
            cv2.putText(frame, f"ID: {id}", (int(x), int(y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) ,2)
            
        
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Demo - Phase 2', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
     
if __name__ == "__main__":
    main()
            