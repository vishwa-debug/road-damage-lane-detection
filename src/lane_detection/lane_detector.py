import cv2
import numpy as np
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_IMAGE = BASE_DIR / "data" / "sample" / "road_test.png"
OUTPUT_DIR = BASE_DIR / "outputs" / "images"

OUTPUT_IMAGE = OUTPUT_DIR / "03_lane_detection.png"
EDGES_IMAGE = OUTPUT_DIR / "01_edges.png"
ROI_IMAGE = OUTPUT_DIR / "02_roi.png"


def detect_lanes(image):
    """
    Detect road lane markings using HLS color filtering
    and Hough Line Transform.
    """

    height, width = image.shape[:2]

    # ---------------------------------------------------------
    # 1. Convert image to HLS
    # ---------------------------------------------------------
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # ---------------------------------------------------------
    # 2. Detect bright / white lane markings
    # ---------------------------------------------------------
    # White markings have:
    # - high lightness
    # - relatively low saturation
    lower_white = np.array([0, 160, 0])
    upper_white = np.array([180, 255, 100])

    white_mask = cv2.inRange(
        hls,
        lower_white,
        upper_white
    )

    # ---------------------------------------------------------
    # 3. Clean small noise
    # ---------------------------------------------------------
    kernel = np.ones((3, 3), np.uint8)

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ---------------------------------------------------------
    # 4. Region of Interest
    # ---------------------------------------------------------
    roi_mask = np.zeros_like(white_mask)

    polygon = np.array([
        [
            (0, height),
            (int(width * 0.38), int(height * 0.55)),
            (int(width * 0.62), int(height * 0.55)),
            (width, height)
        ]
    ], dtype=np.int32)

    cv2.fillPoly(
        roi_mask,
        polygon,
        255
    )

    roi = cv2.bitwise_and(
        white_mask,
        roi_mask
    )

    # ---------------------------------------------------------
    # 5. Edge detection on white markings
    # ---------------------------------------------------------
    edges = cv2.Canny(
        roi,
        50,
        150
    )

    # ---------------------------------------------------------
    # 6. Hough Line Transform
    # ---------------------------------------------------------
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=25,
        maxLineGap=80
    )

    lane_image = image.copy()

    left_candidates = []
    right_candidates = []

    total_lines = 0

    # ---------------------------------------------------------
    # 7. Process Hough lines
    # ---------------------------------------------------------
    if lines is not None:

        lines = np.asarray(lines).reshape(-1, 4)

        total_lines = len(lines)

        for x1, y1, x2, y2 in lines:

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Ignore horizontal lines
            if abs(slope) < 0.30:
                continue

            # Ignore extremely steep noise
            if abs(slope) > 4.0:
                continue

            # Midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Only road area
            if mid_y < height * 0.55:
                continue

            # -------------------------------------------------
            # Estimate x position at bottom of useful area
            # -------------------------------------------------
            y_bottom = height * 0.90

            x_bottom = x1 + (
                (y_bottom - y1) / (y2 - y1)
            ) * (x2 - x1)

            # -------------------------------------------------
            # Left lane candidate
            # -------------------------------------------------
            if slope < 0:

                if (
                    width * 0.20
                    < x_bottom
                    < width * 0.55
                ):
                    left_candidates.append(
                        (x1, y1, x2, y2)
                    )

            # -------------------------------------------------
            # Right lane candidate
            # -------------------------------------------------
            elif slope > 0:

                if (
                    width * 0.60
                    < x_bottom
                    < width * 0.98
                ):
                    right_candidates.append(
                        (x1, y1, x2, y2)
                    )

    print(f"Total Hough lines detected: {total_lines}")
    print(
        f"Left line candidates : {len(left_candidates)}"
    )
    print(
        f"Right line candidates: {len(right_candidates)}"
    )

    # ---------------------------------------------------------
    # 8. Fit lane line
    # ---------------------------------------------------------
    def fit_lane_line(line_group):

        if len(line_group) == 0:
            return None

        slopes = []
        intercepts = []
        weights = []

        for x1, y1, x2, y2 in line_group:

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            intercept = y1 - slope * x1

            length = np.sqrt(
                (x2 - x1) ** 2
                +
                (y2 - y1) ** 2
            )

            slopes.append(slope)
            intercepts.append(intercept)
            weights.append(length)

        if not slopes:
            return None

        slope = np.average(
            slopes,
            weights=weights
        )

        intercept = np.average(
            intercepts,
            weights=weights
        )

        if abs(slope) < 0.01:
            return None

        # Don't extend to the very bottom
        y_bottom = int(height * 0.90)

        # Start near horizon
        y_top = int(height * 0.60)

        x_bottom = int(
            (y_bottom - intercept) / slope
        )

        x_top = int(
            (y_top - intercept) / slope
        )

        return (
            x_bottom,
            y_bottom,
            x_top,
            y_top
        )

    # ---------------------------------------------------------
    # 9. Fit left and right lanes
    # ---------------------------------------------------------
    left_lane = fit_lane_line(left_candidates)
    right_lane = fit_lane_line(right_candidates)

    # ---------------------------------------------------------
    # 10. Draw left lane
    # ---------------------------------------------------------
    if left_lane is not None:

        x1, y1, x2, y2 = left_lane

        cv2.line(
            lane_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

        print("Left lane detected.")

    else:
        print("Left lane not detected.")

    # ---------------------------------------------------------
    # 11. Draw right lane
    # ---------------------------------------------------------
    if right_lane is not None:

        x1, y1, x2, y2 = right_lane

        cv2.line(
            lane_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

        print("Right lane detected.")

    else:
        print("Right lane not detected.")

    return lane_image

def main():

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------
    if not INPUT_IMAGE.exists():

        raise FileNotFoundError(
            f"Input image not found: {INPUT_IMAGE}"
        )

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------
    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:

        raise ValueError(
            "Could not read the input image."
        )

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Run detector
    # ---------------------------------------------------------
    result = detect_lanes(image)

    # ---------------------------------------------------------
    # Save final result
    # ---------------------------------------------------------
    cv2.imwrite(
        str(OUTPUT_IMAGE),
        result
    )

    print()
    print("Lane detection completed successfully.")
    print(f"Input : {INPUT_IMAGE}")
    print(f"Edges : {EDGES_IMAGE}")
    print(f"ROI   : {ROI_IMAGE}")
    print(f"Output: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()