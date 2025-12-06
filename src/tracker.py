import math

class DistanceTracker:
    def __init__(self):
        # dict: {ID_number : (cx, cy)}
        self.center_points = {}
        self.id_dissappeared = {}
        self.max_dissappeared = 50
        self.threshold = 50
        self.id_count = 0
    
    def update(self, objects_rect):
        """
        Input: Boxes from YOLO [[x, y, w , h]]
        Output: ID added boxes [[x, y, w , h, ID]]
        """
        objects_ids = [] # [[x, y, w , h, ID]]
        
        new_center_points = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2 # center x
            cy = (y + y + h) // 2 # center y
            new_center_points.append((cx, cy, x, y, w, h))

            # tracking old ids
            used_existing_ids = []

            
        for new_pt in new_center_points:
            ncx, ncy, nx, ny, nw, nh = new_pt
            same_object_detected = False

            for id, pt in self.center_points.items():
                # distance between new and old points
                dist = math.hypot(ncx - pt[0], ncy - pt[1])

                if dist < self.threshold:
                    self.center_points[id] = (ncx, ncy) # update location
                    self.id_dissappeared[id] = 0 
                    
                    objects_ids.append([nx, ny, nw, nh, id])
                    same_object_detected = True
                    used_existing_ids.append(id)
                    break
            
            # if it's a new car 
            if not same_object_detected:
                self.center_points[self.id_count] = (ncx, ncy)
                self.id_dissappeared[self.id_count] = 0 # has not yet disappeared
                objects_ids.append([nx, ny, nw, nh, self.id_count])
                self.id_count += 1

        # if it is under a tree
        for id in list(self.center_points.keys()):
            if id not in used_existing_ids:
                
                self.id_dissappeared[id] += 1
                
                if self.id_dissappeared[id] > self.max_dissappeared:
                    del self.center_points[id]
                    del self.id_dissappeared[id]
                else:
                    
                    pass

        return objects_ids