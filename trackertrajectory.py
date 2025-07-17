import numpy as np
from collections import defaultdict

class RobustBoxTracker:
    def __init__(self, max_disappeared=30, max_distance=80, reidentification_threshold=0.7,
                 area_weight=0.3, position_weight=0.4, aspect_weight=0.3):
        self.next_id = 0
        self.objects = {}  # Active object centroids
        self.disappeared = {}
        self.object_history = {}
        self.lost_objects = {}

        self.trajectories = defaultdict(list)  # Centroid history per object ID

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.reid_threshold = reidentification_threshold
        self.area_weight = area_weight
        self.position_weight = position_weight
        self.aspect_weight = aspect_weight

        self.total_unique_boxes = 0
        self.reidentifications = 0

    def calculate_box_features(self, bbox):
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'center': center,
            'bbox': bbox
        }

    def calculate_similarity(self, f1, f2):
        area_sim = 1 - abs(f1['area'] - f2['area']) / max(f1['area'], f2['area'], 1)
        aspect_sim = 1 - abs(f1['aspect_ratio'] - f2['aspect_ratio']) / max(f1['aspect_ratio'], f2['aspect_ratio'], 1)
        pos_dist = np.linalg.norm(np.array(f1['center']) - np.array(f2['center']))
        pos_sim = max(0, 1 - pos_dist / 500)
        return (self.area_weight * area_sim +
                self.aspect_weight * aspect_sim +
                self.position_weight * pos_sim)

    def register(self, bbox):
        features = self.calculate_box_features(bbox)
        object_id = self.next_id
        self.objects[object_id] = features['center']
        self.disappeared[object_id] = 0
        self.object_history[object_id] = {
            'features': [features],
            'last_seen': 0,
            'total_frames': 1
        }
        self.trajectories[object_id].append(features['center'])
        self.next_id += 1
        self.total_unique_boxes += 1
        return object_id

    def deregister(self, object_id):
        if object_id in self.objects:
            self.lost_objects[object_id] = {
                'history': self.object_history[object_id],
                'last_position': self.objects[object_id],
                'frames_lost': 0
            }
            del self.objects[object_id]
            del self.disappeared[object_id]

    def try_reidentification(self, new_features):
        best_match_id = None
        best_similarity = 0
        for lost_id, data in list(self.lost_objects.items()):
            if data['frames_lost'] > self.max_disappeared * 2:
                del self.lost_objects[lost_id]
                continue
            similarities = [
                self.calculate_similarity(new_features, f)
                for f in data['history']['features'][-5:]
            ]
            avg_sim = np.mean(similarities)
            if avg_sim > best_similarity and avg_sim > self.reid_threshold:
                best_similarity = avg_sim
                best_match_id = lost_id
        return best_match_id, best_similarity

    def update(self, rects, frame_number=0):
        for lost_id in self.lost_objects:
            self.lost_objects[lost_id]['frames_lost'] += 1

        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.get_objects()

        input_features = [self.calculate_box_features(r) for r in rects]
        input_centroids = np.array([f['center'] for f in input_features])

        if len(self.objects) == 0:
            for i, f in enumerate(input_features):
                reid, sim = self.try_reidentification(f)
                if reid is not None:
                    self.objects[reid] = f['center']
                    self.disappeared[reid] = 0
                    self.object_history[reid]['features'].append(f)
                    self.object_history[reid]['last_seen'] = frame_number
                    self.object_history[reid]['total_frames'] += 1
                    self.trajectories[reid].append(f['center'])
                    del self.lost_objects[reid]
                    self.reidentifications += 1
                else:
                    self.register(rects[i])
            return self.get_objects()

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids, axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols or D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            self.object_history[object_id]['features'].append(input_features[col])
            self.object_history[object_id]['last_seen'] = frame_number
            self.object_history[object_id]['total_frames'] += 1
            self.trajectories[object_id].append(input_centroids[col])
            if len(self.object_history[object_id]['features']) > 10:
                self.object_history[object_id]['features'].pop(0)
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            f = input_features[col]
            reid, sim = self.try_reidentification(f)
            if reid is not None:
                self.objects[reid] = f['center']
                self.disappeared[reid] = 0
                self.object_history[reid]['features'].append(f)
                self.object_history[reid]['last_seen'] = frame_number
                self.object_history[reid]['total_frames'] += 1
                self.trajectories[reid].append(f['center'])
                del self.lost_objects[reid]
                self.reidentifications += 1
            else:
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

    def get_trajectories(self):
        return self.trajectories
