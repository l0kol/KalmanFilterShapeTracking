# Shape Tracking System

## Overview

This system tracks simple shapes in a video feed. It maintains a record of shapes' previous positions and visualizes their paths and bounding boxes on the video frames.

The system consists of two primary components:
* **Shape Detection**: Identifies shapes in each frame based on contours of threshold frame. Thresholding is done on a grayscale image of the scene and on a HSV image of the scene.
* **Shape Tracking**: Matches detected shapes across frames using position and color similarity, and predicts positions for missing shapes using opencv's implementation of Kalman filter.


### Measured performance:
* Average time per frame: 16 milliseconds

Performance was measured on the following system:
* CPU: AMD Ryzen 7 6800H
* Integrated GPU: AMD Radeon Graphics
* RAM: 16 GB

## Usage

Example usage of the shape tracking system:

```python
cap = cv2.VideoCapture("video.mp4")

frameShape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# Initialize the shape tracker with the frame shape
tracker = ShapeTracker(frameShape)

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect shapes
    detected_shapes = tracker.detect_shapes(frame)

    # Track shapes
    tracker.track_shapes(detected_shapes)
    
    # Draw the bounding boxes
    tracker.draw_bounding_boxes(frame)

    # Draw the tracked paths
    tracker.draw_paths(frame)

    # Show the frame
    cv2.imshow("Shape Tracking", frame)

    # Exit if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break
```

## How the shape tracking works
#### 1. Shape Detection

Shapes are detected in each frame by:
* Grayscale Thresholding:
    * The grayscale image undergoes Gaussian blur and adaptive thresholding.
        Contours are then extracted from the binary image.
* Color Thresholding:
    * For tracked shapes, HSV color thresholds are applied to filter the frame.

Both methods use morphological operations to clean the binary images.

Each detected shape is characterized by:
- Center: The centroid of the shape (cx, cy).
- Bounding Box: The rectangle enclosing the shape (x, y, width, height).
- Perimeter: Contour perimeter.
- Area: Contour area.
- HSV Color: The HSV color value at the shape's center.
- Shape Type: Classified as "SQUARE", "CIRCLE", or "OTHER" based on geometry.

### 2. Shape Registration

New shapes are registered to the tracker when they cannot be matched with any existing shape. A unique ID is assigned to the new shape and a shape object is created to store its properties.

### 3. Shape Matching

Detected shapes in the current frame are matched with previously tracked shapes using:
* Euclidean Distance: Between the centers of the detected shape and the tracked shape.
* Color Similarity: Difference in HSV color space.

If a match is found:
* The tracked shape's properties (position, bounding box, etc.) are updated.
    Its path is extended with the new center.

If no match is found:
* The shape is registered as a new tracked shape.

### 4. Missing Shapes

For shapes not detected in the current frame:
* Kalman Filter Prediction: Predicts the next position based on previous positions.
* If a shape remains missing for too many frames, it is removed from tracking.

### 5. Visualization
* Paths are drawn on the video frames to visualize the movement of shapes.
* Bounding boxes are drawn around the shapes with their IDs and types (circle or square).

## Improvements
The detection of overlapping shapes could be improved by adding some additional logic to the shape detection algorithm. For example utilising the area, aspect ratio, approximated shape, etc. to determine if two shapes are overlapping. This data is already being collected, so it could be used to improve the system.

Different shape detection methods could be used to improve the system. For example, edge detetection with Canny or Sobel filters could be used instead of thresholding the grayscale image, possibly improving the performance.



