import cv2
import numpy as np
import random


class Shape:
    """
    A class to represent a geometric shape and track its movement using a Kalman filter.
    Attributes:
    -----------
    shape_id : int
        Unique identifier for the shape.
    center : tuple
        The (x, y) coordinates of the shape's center.
    area : float
        The area of the shape.
    perimeter : float
        The perimeter of the shape.
    path : list
        A list of (x, y) coordinates representing the path of the shape.
    missing_frames : int
        Counter for the number of frames the shape has been missing.
    bbox : tuple
        The bounding box of the shape.
    shape_color : tuple
        The color of the shape.
    type : str
        The type of the shape (e.g., circle, square).
    trace_color : tuple or None
        The color used to trace the shape's path.
    kf : cv2.KalmanFilter
        The Kalman filter used for tracking the shape.
    Methods:
    --------
    update(new_center):
        Update the Kalman filter with a new center.
    predict():
        Predict the next position of the shape using the Kalman filter.
    get_bbox():
        Generate the bounding box based on the current position.
    """

    def __init__(self, shape_id, center, area, perimeter, bbox, color, shape_type):
        self.shape_id = shape_id
        self.center = center
        self.area = area
        self.perimeter = perimeter
        self.path = [center]  # Store the path of the shape
        self.missing_frames = 0
        self.bbox = bbox
        self.shape_color = color
        self.type = shape_type
        self.trace_color = None

        # (x, y, dx, dy) for 4 states, (x, y) for 2 measurements
        self.kf = cv2.KalmanFilter(4, 2)
        # Transition matrix, linear motion model
        self.kf.transitionMatrix = np.array(
            [[1, 0, 0.1, 0], [0, 1, 0, 1], [0, 0, 0.1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        # Directly observe the position
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        # Uncertainty in the process
        self.kf.processNoiseCov = np.array(
            [[1e-1, 0, 0, 0], [0, 1e-1, 0, 0], [0, 0, 1e-2, 0], [0, 0, 0, 1e-2]],
            dtype=np.float32,
        )
        # Uncertainty in the measurements
        self.kf.measurementNoiseCov = np.array([[1e-1, 0], [0, 1e-1]], dtype=np.float32)

        # Initial update with the starting center position
        self.kf.predict()
        self.kf.correct(np.array([center[0], center[1]], dtype=np.float32))

    def update(self, new_center):
        """Update the Kalman filter with a new center."""
        self.kf.predict()
        self.kf.correct(np.array([new_center[0], new_center[1]], dtype=np.float32))
        self.center = new_center
        self.path.append(new_center)
        self.missing_frames = 0  # Reset missing frames

    def predict(self):
        """Predict the next position of the shape using Kalman filter."""
        self.kf.predict()
        predicted_center = (int(self.kf.statePost[0, 0]), int(self.kf.statePost[1, 0]))
        self.center = predicted_center

    def get_bbox(self):
        """Generate the bounding box based on the current position."""
        x, y = self.center
        w = self.bbox[2]
        h = self.bbox[3]
        return x - w // 2, y - h // 2, w, h


class ShapeTracker:
    def __init__(self, frameShape):
        self.frameHeight: int = frameShape[0]
        self.frameWidth: int = frameShape[1]
        self.tracked_shapes: Shape = {}
        self.tracked_colors: list = {}
        self.next_shape_id: int = 0

    @staticmethod
    def _random_color() -> tuple:
        """
        Generate a random RGB color.
        Returns:
            tuple: A tuple containing three integers representing an RGB color.
        """
        return tuple(random.randint(0, 255) for _ in range(3))

    @staticmethod
    def _calculate_hsv_bounds(color, tolerance):
        """Calculate HSV bounds for color thresholding."""
        color = color.astype(np.uint16)
        lower_hue = ((int(color[0]) - tolerance) % 180) % 180
        upper_hue = (color[0] + tolerance) % 180
        lower_saturation = max(color[1] - tolerance, 0)
        upper_saturation = min(color[1] + tolerance, 255)
        lower_value = max(color[2] - tolerance, 0)
        upper_value = min(color[2] + tolerance, 255)
        lower_bound = np.array(
            [lower_hue, lower_saturation, lower_value], dtype=np.uint8
        )
        upper_bound = np.array(
            [upper_hue, upper_saturation, upper_value], dtype=np.uint8
        )

        return lower_bound, upper_bound

    def _threshold_colors(self, frame, frame_gray) -> list:
        """
        Thresholds colors in the given frame based on the detected shapes colors.
        Args:
            frame (numpy.ndarray): The input frame in BGR color space.
            frame_gray (numpy.ndarray): The grayscale version of the input frame.
            tolerance (int): The tolerance value for color thresholding.
        Returns:
            tuple: A tuple containing the modified grayscale frame and a list of contours.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contours = []
        for color in self.tracked_colors.values():
            lower, upper = color["bounds"]
            mask = cv2.inRange(hsv_frame, lower, upper)

            # Dialate the mask to remove noise around the shape
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.dilate(mask, kernel, iterations=1)
            # Find contours
            _contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours.extend(_contours)

            # Subtract the mask from the gray_img, only remove the detected shapes
            frame_gray = cv2.bitwise_and(
                frame_gray, frame_gray, mask=cv2.bitwise_not(cleaned_mask)
            )

        return frame_gray, contours

    def _threshold_grayscale(self, frame_gray) -> list:
        """
        Applies a series of image processing techniques to a grayscale frame to extract contours.

        Args:
            frame_gray (numpy.ndarray): The input grayscale image.
        Returns:
            list: A list of contours found in the processed binary image.
        """
        blurred_img = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        # Threshold the image
        binary_img = cv2.adaptiveThreshold(
            blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_binary = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

        # https://stackoverflow.com/questions/66924925/how-can-i-remove-double-lines-detected-along-the-edges
        cv2.floodFill(
            cleaned_binary, mask=None, seedPoint=(int(0), int(0)), newVal=(255)
        )

        cv2.floodFill(cleaned_binary, mask=None, seedPoint=(int(0), int(0)), newVal=(0))

        output_path = "floodFill.png"
        cv2.imwrite(output_path, cleaned_binary)
        print(f"Saved contours image to: {output_path}")

        # Find contours
        contours, _ = cv2.findContours(
            cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def detect_shapes(self, frame) -> list:
        """
        Detects shapes in a given frame.
        This method processes the input frame to detect various shapes based on their contours,
        and colors. It returns a list of detected shapes with their properties.
        Args:
            frame (numpy.ndarray): The input image frame in which shapes are to be detected.
        Returns:
            list: A list of dictionaries, each containing the following keys:
                - "bbox" (tuple): Bounding box of the shape (x, y, width, height).
                - "contour" (numpy.ndarray): Contour of the detected shape.
                - "center" (tuple): Center coordinates (cx, cy) of the shape.
                - "area" (float): Area of the shape.
                - "perimeter" (float): Perimeter of the shape.
                - "color_hsv" (numpy.ndarray): HSV color of the shape.
                - "shape_type" (str): Type of the shape ('SQUARE', 'CIRCLE', or 'OTHER').
        """
        detected_shapes = []
        contours = []

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(self.tracked_colors) > 0:
            # If we have detected shapes, threshold the image based on the shapes colors
            frame_gray, contours = self._threshold_colors(frame, frame_gray)

        contours.extend(self._threshold_grayscale(frame_gray))

        for contour in contours:
            # Check if the contour area is big enough
            if cv2.contourArea(contour) < 1700:
                continue

            # Compute moments for the contour
            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            area = M["m00"]
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate circularity
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0

            # Classify the shape
            if len(cv2.approxPolyDP(contour, 0.02 * perimeter, True)) == 4:
                shape_type = "SQUARE"  # It's a square
            elif circularity > 0.8:
                shape_type = "CIRCLE"  # It's a circle
            else:
                shape_type = "OTRH"  # All other shapes

            # Get color of the shape, take center pixel
            cx = x + w // 2
            cy = y + h // 2
            color = frame[cy, cx]
            # Get the HSV color
            color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]

            detected_shapes.append(
                {
                    "bbox": (x, y, w, h),
                    "contour": contour,
                    "center": (cx, cy),
                    "area": area,
                    "perimeter": perimeter,
                    "color_hsv": color_hsv,
                    "shape_type": shape_type,
                }
            )

        return detected_shapes

    def track_shapes(self, detected_shapes):
        """
        Tracks and updates shapes based on detected shapes in the current frame.

        Args:
            detected_shapes (list): List of dictionaries with detected shape attributes:
                - "center" (tuple): Shape's center (x, y).
                - "perimeter" (float): Perimeter of the shape.
                - "bbox" (list): Bounding box as [x, y, width, height].
                - "color_hsv" (numpy.ndarray): HSV color of the shape.
                - "area" (float): Shape's area.
                - "shape_type" (str): Shape type (e.g., "circle", "rectangle").

        Updates:
            - self.tracked_shapes: Maps shape IDs to Shape objects.
            - self.tracked_colors: Maps shape IDs to their HSV colors.
            - self.next_shape_id: Increments for newly detected shapes.
        """
        new_tracked_shapes = {}
        unmatched_tracked_shapes: Shape = self.tracked_shapes.copy()

        for shape in detected_shapes:
            cx, cy = shape["center"]
            matched_id = None

            # Try to match this shape with existing tracked shapes
            for shape_id, shape_obj in self.tracked_shapes.items():
                # Calculate the Euclidean distance between the centers of the shapes
                dist = np.sqrt(
                    (cx - shape_obj.center[0]) ** 2 + (cy - shape_obj.center[1]) ** 2
                )

                # Check color difference in hsv space
                tracked_color = self.tracked_colors[shape_id]["hsv"].astype(np.int16)
                color_diff = np.linalg.norm(
                    shape["color_hsv"].astype(np.int16) - tracked_color
                )

                if dist < 300 and color_diff < 10:
                    matched_id = shape_id
                    break

            if matched_id is not None:
                # Update the Kalman filter and tracked shape
                shape_obj = self.tracked_shapes[matched_id]
                shape_obj.update((cx, cy))
                shape_obj.perimeter = shape["perimeter"]
                shape_obj.bbox = list(shape["bbox"])
                new_tracked_shapes[matched_id] = shape_obj

                del self.tracked_shapes[matched_id]
                del unmatched_tracked_shapes[matched_id]
            else:
                # Assign new ID and create a new shape object with Kalman filter
                self.next_shape_id += 1
                new_shape = Shape(
                    self.next_shape_id,
                    (cx, cy),
                    shape["area"],
                    shape["perimeter"],
                    list(shape["bbox"]),
                    shape["color_hsv"],
                    shape["shape_type"],
                )
                new_shape.trace_color = self._random_color()
                new_tracked_shapes[self.next_shape_id] = new_shape

                # Store HSV color and precomputed bounds
                lower_bound, upper_bound = self._calculate_hsv_bounds(
                    shape["color_hsv"], 4
                )
                self.tracked_colors[self.next_shape_id] = {
                    "hsv": shape["color_hsv"],
                    "bounds": (lower_bound, upper_bound),
                }

        # Handle missing shapes (objects not detected in the current frame)
        for shape_id, shape_obj in unmatched_tracked_shapes.items():
            # If the shape is not at the edge of the frame, allow more missing frames
            max_missing_frames = 5 if self._is_at_edge(shape_obj.bbox) else 15

            if shape_obj.missing_frames < max_missing_frames:  #
                if (
                    len(shape_obj.path) > 5
                ):  # Kalman filter need a few frames to predict the position, otherwise it will be too inaccurate
                    shape_obj.predict()  # Get predicted position using Kalman filter
                else:
                    shape_obj.update(
                        shape_obj.center
                    )  # Update the shape with the same position if we don't have enough frames
                shape_obj.missing_frames += 1
                new_tracked_shapes[shape_id] = shape_obj

            else:
                # Remove the shape if it's missing for too many frames
                del self.tracked_colors[shape_id]

        self.tracked_shapes = new_tracked_shapes

    def draw_paths(self, frame):
        """
        Draws the paths of tracked shapes on the given frame.
        Args:
            frame (numpy.ndarray): The image frame on which to draw the paths.
        """
        for _, shape in self.tracked_shapes.items():
            path = shape.path
            color = tuple(map(int, shape.trace_color))

            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], color, 4)

            cx, cy = shape.center
            cv2.circle(frame, (cx, cy), 5, color, -1)

    def draw_bounding_boxes(self, frame):
        """
        Draws bounding boxes around tracked shapes on the given frame.
        Args:
            frame (numpy.ndarray): The frame on which to draw the bounding boxes.

        The bounding box and text are drawn using OpenCV functions:
        - cv2.rectangle: Draws the rectangle around the bounding box.
        - cv2.putText: Writes the shape ID and type above the bounding box.
        """
        for shape_id, shape in self.tracked_shapes.items():
            if shape.missing_frames == 0:
                x, y, w, h = shape.bbox
            else:
                x, y, w, h = shape.get_bbox()

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"[{shape_id}] {shape.type}",
                (x, y - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 255, 0),
                2,
            )

    def _is_at_edge(self, bbox, edge_margin=3) -> bool:
        """
        Check if a bounding box is at the edge of the frame.

        Args:
            bbox (tuple): A tuple (x, y, w, h) representing the bounding box coordinates and dimensions.
            edge_margin (int, optional): The margin from the edge to consider. Defaults to 3.

        Returns:
            bool: True if the bounding box is at the edge of the frame, False otherwise.
        """
        x, y, w, h = bbox
        return (
            x <= edge_margin
            or y <= edge_margin
            or x + w >= self.frameWidth - edge_margin
            or y + h >= self.frameHeight - edge_margin
        )


if __name__ == "__main__":
    cap = cv2.VideoCapture("luxonis_task_video.mp4")
    frameShape = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    tracker = ShapeTracker(frameShape)

    allTimes = np.empty(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Detect shapes
        detected_shapes = tracker.detect_shapes(frame)

        # Track shapes
        tracker.track_shapes(detected_shapes)

        tracker.draw_bounding_boxes(frame)

        # Draw the tracked paths
        tracker.draw_paths(frame)

        # Show the frame
        cv2.imshow("Shape Tracking", frame)

        # Exit if 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
