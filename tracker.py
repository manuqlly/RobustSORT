import numpy as np

class RobustBoxTracker:
    def __init__(self, max_disappeared=30, max_distance=80, reidentification_threshold=0.7,
                 area_weight=0.3, position_weight=0.4, aspect_weight=0.3):
        self.next_id = 0
        self.objects = {}  # Current active objects
        self.disappeared = {}  # How long objects have been gone
        self.object_history = {}  # Historical data for re-identification
        self.lost_objects = {}  # Recently lost objects for re-identification
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.reid_threshold = reidentification_threshold
        self.area_weight = area_weight
        self.position_weight = position_weight
        self.aspect_weight = aspect_weight
        
        # Statistics
        self.total_unique_boxes = 0
        self.reidentifications = 0

    def calculate_box_features(self, bbox):
        """Calculate features for box comparison"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'center': (center_x, center_y),
            'width': width,
            'height': height,
            'bbox': bbox
        }

    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two boxes based on multiple features"""
        # Area similarity (normalized)
        area1, area2 = features1['area'], features2['area']
        area_sim = 1 - abs(area1 - area2) / max(area1, area2, 1)
        
        # Aspect ratio similarity
        ar1, ar2 = features1['aspect_ratio'], features2['aspect_ratio']
        aspect_sim = 1 - abs(ar1 - ar2) / max(ar1, ar2, 1)
        
        # Position similarity (normalized by frame dimensions - assuming 1920x1080)
        pos1, pos2 = features1['center'], features2['center']
        pos_distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        pos_sim = max(0, 1 - pos_distance / 500)  # Normalize by expected max distance
        
        # Combined similarity
        total_sim = (self.area_weight * area_sim + 
                    self.aspect_weight * aspect_sim + 
                    self.position_weight * pos_sim)
        
        return total_sim

    def register(self, bbox):
        """Register a new object"""
        features = self.calculate_box_features(bbox)
        self.objects[self.next_id] = features['center']
        self.disappeared[self.next_id] = 0
        self.object_history[self.next_id] = {
            'features': [features],
            'last_seen': 0,  # frame number
            'total_frames': 1
        }
        self.next_id += 1
        self.total_unique_boxes += 1
        return self.next_id - 1

    def deregister(self, object_id):
        """Move object to lost objects for potential re-identification"""
        if object_id in self.objects:
            # Keep in lost objects for potential re-identification
            self.lost_objects[object_id] = {
                'history': self.object_history[object_id],
                'last_position': self.objects[object_id],
                'frames_lost': 0
            }
            
            del self.objects[object_id]
            del self.disappeared[object_id]

    def try_reidentification(self, new_features):
        """Try to re-identify a new detection with lost objects"""
        best_match_id = None
        best_similarity = 0
        
        for lost_id, lost_data in list(self.lost_objects.items()):
            # Don't try to re-identify very old lost objects
            if lost_data['frames_lost'] > self.max_disappeared * 2:
                del self.lost_objects[lost_id]
                continue
            
            # Calculate similarity with historical features
            similarities = []
            for hist_features in lost_data['history']['features'][-5:]:  # Use last 5 features
                sim = self.calculate_similarity(new_features, hist_features)
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity and avg_similarity > self.reid_threshold:
                best_similarity = avg_similarity
                best_match_id = lost_id
        
        return best_match_id, best_similarity

    def update(self, rects, frame_number=0):
        """Update tracker with new detections"""
        # Increment frames lost for lost objects
        for lost_id in self.lost_objects:
            self.lost_objects[lost_id]['frames_lost'] += 1
        
        if len(rects) == 0:
            # No detections - mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_objects()

        # Calculate features for all new detections
        input_features = []
        input_centroids = []
        for rect in rects:
            features = self.calculate_box_features(rect)
            input_features.append(features)
            input_centroids.append(features['center'])

        input_centroids = np.array(input_centroids)

        if len(self.objects) == 0:
            # No existing objects - try re-identification first
            registered_ids = []
            for i, features in enumerate(input_features):
                reident_id, similarity = self.try_reidentification(features)
                if reident_id is not None:
                    # Re-identified object
                    self.objects[reident_id] = features['center']
                    self.disappeared[reident_id] = 0
                    self.object_history[reident_id]['features'].append(features)
                    self.object_history[reident_id]['last_seen'] = frame_number
                    self.object_history[reident_id]['total_frames'] += 1
                    registered_ids.append(reident_id)
                    del self.lost_objects[reident_id]
                    self.reidentifications += 1
                else:
                    # New object
                    new_id = self.register(rects[i])
                    registered_ids.append(new_id)
            
            return {id: self.objects.get(id, input_centroids[i]) for i, id in enumerate(registered_ids)}
        
        else:
            # Existing objects - perform matching
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Calculate distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Hungarian algorithm approximation
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Match existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update history
                self.object_history[object_id]['features'].append(input_features[col])
                self.object_history[object_id]['last_seen'] = frame_number
                self.object_history[object_id]['total_frames'] += 1
                
                # Keep only recent features
                if len(self.object_history[object_id]['features']) > 10:
                    self.object_history[object_id]['features'].pop(0)
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched existing objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                # More existing objects than detections
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Handle unmatched detections - try re-identification first
            for col in unused_col_indices:
                features = input_features[col]
                reident_id, similarity = self.try_reidentification(features)
                
                if reident_id is not None:
                    # Re-identified object
                    self.objects[reident_id] = features['center']
                    self.disappeared[reident_id] = 0
                    self.object_history[reident_id]['features'].append(features)
                    self.object_history[reident_id]['last_seen'] = frame_number
                    self.object_history[reident_id]['total_frames'] += 1
                    del self.lost_objects[reident_id]
                    self.reidentifications += 1
                else:
                    # New object
                    self.register(rects[col])
        
        return self.get_objects()

    def get_objects(self):
        return self.objects
    
    def get_stats(self):
        return {
            'total_unique_boxes': self.total_unique_boxes,
            'reidentifications': self.reidentifications,
            'active_objects': len(self.objects),
            'lost_objects': len(self.lost_objects)
        }
