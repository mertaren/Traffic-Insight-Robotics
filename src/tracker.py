import math
import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        # dynamParams: x_new, y_new, v_x, v_y
        # measureParams: x, y -from YOLO-
        self.kf = cv2.KalmanFilter(4, 2)
        # delta t
        self.dt = 1

        self.kf.transitionMatrix = np.array([[1, 0 , self.dt, 0],
                                            [0, 1, 0 ,self.dt],
                                            [0, 0 ,1 ,0],
                                            [0, 0, 0, 1]], np.float32)

        self.kf.measurementMatrix = np.array([[1, 0 ,0 ,0],
                                              [0, 1, 0, 0]], np.float32)

        # noise params
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    def predict(self):
            
            prediction = self.kf.predict()
            return prediction[0], prediction[1]
        
    def correct(self, x, y ):
            
            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            self.kf.correct(measurement)
            return self.kf.statePost[0], self.kf.statePost[1]    


class Tracker:
      def __init__(self):
          # {ID : {'kf': KalmanObject, 'bbox' : [x,y,w,h], 'missing': 0}}
          self.tracks = {}
          self.track_id_count = 0
          self.dist_th = 100 # threshold
          self.max_frame_missing = 30

      def update(self, detections):
            """
            detections :[x, y, w, h] list from YOLO
            """
            
            # predict the new location of all registered vehicles
            for track_id in self.tracks:
                kf = self.tracks[track_id]['kf']
                pred_x, pred_y = kf.predict()

                old_w, old_h = self.tracks[track_id]['bbox'][2], self.tracks[track_id]['bbox'][3]
                self.tracks[track_id]['bbox'] = [pred_x - old_w/2, pred_y - old_h/2, old_w, old_h]

            used_track_ids = []
            assigned_det_indices = []

            # find the center of boxes
            det_centers = []
            for det in detections:
                x, y, w ,h = det
                det_centers.append((x + w/2, y + h/2))
            
            # compare the new boxes
            for track_id, track_data in self.tracks.items():
                 # estimated center of the vehicle
                 tx = track_data['bbox'][0] + track_data['bbox'][2]/2
                 ty = track_data['bbox'][1] + track_data['bbox'][3]/2
                
                 best_dist = self.dist_th
                 best_det_idx = -1

                 for idx ,(dx, dy) in enumerate(det_centers):
                      if idx in assigned_det_indices: continue

                      dist = math.hypot(tx -dx, ty - dy)

                      if dist < best_dist:
                           best_dist = dist
                           best_det_idx = idx
                
            # correction
            if best_det_idx != -1:
                 # update kalman with real data
                 dx, dy = det_centers[best_det_idx]
                 track_data['kf'].correct(dx, dy)

                 # update
                 self.tracks[track_id]['bbox'] = detections[best_det_idx]
                 self.tracks[track_id]['missing'] = 0

                 used_track_ids.append(track_id)
                 assigned_det_indices.append(best_det_idx)
            else:
                 # no matching, vehicle missing
                 self.tracks[track_id]['missing'] += 1




