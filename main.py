import cv2
from src.detector import VehicleDetector

VIDEO_PATH = "data/videos/testvideo.mp4"

def main():
    detector = VehicleDetector()

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
        
        frame = cv2.resize(frame, (1280, 720))

        detections = detector.detect(frame)
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det

            cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)
            
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) ,2 )
            
        cv2.imshow('Demo - Phase 1', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
     
if __name__ == "__main__":
    main()
            