# RobustSort

RobustSort is a lightweight object tracking algorithm that matches bounding boxes using area, position, and aspect ratio similarity. It supports re-identification of lost objects, making it ideal for real-time video tracking in low-resource environments without deep learning or Kalman filters.

[![Demo Video](https://img.shields.io/badge/Watch-Demo-red)](RobustSort.mp4)
[![Trajectory Demo](https://img.shields.io/badge/Watch-Trajectory_Demo-blue)](RobustSortTrajectory.mp4)

## Features

- **Multi-feature Matching**: Uses box area, aspect ratio, and centroid position to match objects
- **Object Re-identification**: Remembers lost objects and can re-identify them when they reappear
- **Trajectory Tracking**: Optional trajectory visualization to track object movement paths
- **No Deep Learning for Tracking**: Only uses geometric features for the actual tracking algorithm
- **Minimal Dependencies**: Core algorithm only requires NumPy
- **Configurable Parameters**: Easily tune the algorithm for your specific use case

## How It's Better Than SORT

RobustSort improves upon the original SORT (Simple Online Realtime Tracking) algorithm in several ways:

1. **No Kalman Filter**: Operates without complex motion prediction, making it more robust to erratic movements
2. **Multi-feature Matching**: Uses not just position but also area and aspect ratio for more robust tracking
3. **Re-identification**: Can remember and re-identify objects that temporarily disappear
4. **Configurable Weights**: Easily adjust which features matter more for your specific use case
5. **Trajectory Visualization**: Built-in support for visualizing object paths

## Installation

No installation required! Just include the tracker files in your project.

```bash
git clone https://github.com/manuqlly/RobustSort.git
cd RobustSort
```

### Requirements

- Python 3.6+
- NumPy
- OpenCV (for demo)
- Ultralytics YOLO (for demo)

Install the dependencies:

```bash
pip install numpy opencv-python ultralytics
```

## Usage

### Basic Usage

```python
from tracker import RobustBoxTracker

# Initialize the tracker
tracker = RobustBoxTracker(
    max_disappeared=30,  # Maximum frames an object can disappear before being deregistered
    max_distance=80,     # Maximum pixel distance for matching
    reidentification_threshold=0.7,  # Minimum similarity for re-identification
    area_weight=0.3,     # Weight for area similarity
    position_weight=0.4, # Weight for position similarity
    aspect_weight=0.3    # Weight for aspect ratio similarity
)

# Update tracker with new detections (list of bounding boxes in format [x1, y1, x2, y2])
objects = tracker.update(detections, frame_number=frame_count)

# objects contains the current tracked objects with their IDs
for object_id, centroid in objects.items():
    print(f"Object ID {object_id} at position {centroid}")
```

### With Trajectory Tracking

```python
from trackertrajectory import RobustBoxTracker

# Initialize and update tracker as before
tracker = RobustBoxTracker()
objects = tracker.update(detections)

# Get trajectories to visualize paths
trajectories = tracker.get_trajectories()
```

### Demo Script

The repository includes two demo scripts:

1. `Demo_without_trajectory.py` - Basic object tracking with ID display
2. `Demo_with_trajectory.py` - Object tracking with movement path visualization

To run the demos:

```bash
# Make sure you have an input.mp4 file in the same directory
python Demo_with_trajectory.py
```

## Parameters

You can configure RobustSort with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_disappeared` | Max frames an object can disappear before deregistration | 30 |
| `max_distance` | Max pixel distance for considering a match | 80 |
| `reidentification_threshold` | Minimum similarity for re-identification | 0.7 |
| `area_weight` | Weight for area similarity in matching | 0.3 |
| `position_weight` | Weight for position similarity in matching | 0.4 |
| `aspect_weight` | Weight for aspect ratio similarity in matching | 0.3 |

## Algorithm Details

RobustSort works through these steps:

1. **Feature Calculation**: Extract centroid position, area, and aspect ratio from each box
2. **Similarity Computation**: Calculate weighted similarity between boxes based on features
3. **Box Matching**: Match existing tracked objects to new detections
4. **Re-identification**: Try to re-identify previously lost objects
5. **Tracking Update**: Update object positions and handle disappearances

## Demo Videos

The repository includes demo videos showing the algorithm in action:

- `RobustSort.mp4` - Shows basic tracking with ID labels
- `RobustSortTrajectory.mp4` - Shows tracking with movement paths

## License

This project is available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Author

RobustSort was created by manuqlly.
