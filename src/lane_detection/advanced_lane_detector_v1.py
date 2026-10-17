import cv2
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_IMAGE = BASE_DIR / "data" / "sample" / "road_test.png"
OUTPUT_IMAGE = BASE_DIR / "outputs" / "images" / "advanced_lane_detection.png"
BIRDSEYE_IMAGE = BASE_DIR / "outputs" / "images" / "birdseye_view.png"


def create_white_mask(image):
    """
    Detect bright/white lane markings using HLS color space.
    """

    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # HLS:
    # H = Hue
    # L = Lightness
    # S = Saturation

    lower_white = np.array([0, 150, 0])
    upper_white = np.array([180, 255, 160])

    mask = cv2.inRange(
        hls,
        lower_white,
        upper_white
    )

    return mask

def perspective_transform(image):
    """
    Convert the road region into a bird's-eye view.

    Coordinates are calculated from the actual image size,
    so the detector works with different resolutions.
    """

    height, width = image.shape[:2]

    # ---------------------------------------------------------
    # Source points
    # ---------------------------------------------------------
    # These are proportional to the current road image.
    #
    #             horizon
    #          /           \
    #         /             \
    #        /               \
    #       /                 \
    #      /                   \
    #
    # The selected area focuses on the road.
    # ---------------------------------------------------------

    src = np.float32([
        [width * 0.38, height * 0.57],
        [width * 0.62, height * 0.57],
        [width * 0.95, height * 0.93],
        [width * 0.08, height * 0.93]
    ])

    # ---------------------------------------------------------
    # Destination points
    # ---------------------------------------------------------

    dst = np.float32([
        [width * 0.25, 0],
        [width * 0.75, 0],
        [width * 0.75, height],
        [width * 0.25, height]
    ])

    # ---------------------------------------------------------
    # Perspective transformation matrix
    # ---------------------------------------------------------

    matrix = cv2.getPerspectiveTransform(
        src,
        dst
    )

    # ---------------------------------------------------------
    # Apply transformation
    # ---------------------------------------------------------

    birdseye = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return birdseye, matrix

def find_lane_pixels(binary_image):
    """
    Find lane pixels using a histogram and sliding windows.
    """

    height, width = binary_image.shape

    # Histogram of lower half of image
    histogram = np.sum(
        binary_image[height // 2:, :],
        axis=0
    )

    midpoint = width // 2

    left_base = np.argmax(
        histogram[:midpoint]
    )

    right_base = np.argmax(
        histogram[midpoint:]
    ) + midpoint

    # Sliding-window parameters
    number_windows = 9
    window_height = height // number_windows

    margin = 80
    minimum_pixels = 30

    nonzero = binary_image.nonzero()

    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_current = left_base
    right_current = right_base

    left_indices = []
    right_indices = []

    for window in range(number_windows):

        win_y_low = (
            height
            - (window + 1) * window_height
        )

        win_y_high = (
            height
            - window * window_height
        )

        left_x_low = left_current - margin
        left_x_high = left_current + margin

        right_x_low = right_current - margin
        right_x_high = right_current + margin

        good_left = (
            (nonzero_y >= win_y_low)
            &
            (nonzero_y < win_y_high)
            &
            (nonzero_x >= left_x_low)
            &
            (nonzero_x < left_x_high)
        ).nonzero()[0]

        good_right = (
            (nonzero_y >= win_y_low)
            &
            (nonzero_y < win_y_high)
            &
            (nonzero_x >= right_x_low)
            &
            (nonzero_x < right_x_high)
        ).nonzero()[0]

        left_indices.append(good_left)
        right_indices.append(good_right)

        if len(good_left) > minimum_pixels:
            left_current = int(
                np.mean(
                    nonzero_x[good_left]
                )
            )

        if len(good_right) > minimum_pixels:
            right_current = int(
                np.mean(
                    nonzero_x[good_right]
                )
            )

    left_indices = np.concatenate(left_indices)
    right_indices = np.concatenate(right_indices)

    left_x = nonzero_x[left_indices]
    left_y = nonzero_y[left_indices]

    right_x = nonzero_x[right_indices]
    right_y = nonzero_y[right_indices]

    return (
        left_x,
        left_y,
        right_x,
        right_y
    )


def fit_lane_polynomial(
    left_x,
    left_y,
    right_x,
    right_y,
    image_shape
):
    """
    Fit second-degree polynomial curves
    to detected lane pixels.
    """

    height = image_shape[0]

    left_fit = None
    right_fit = None

    if len(left_x) > 100:

        left_fit = np.polyfit(
            left_y,
            left_x,
            2
        )

    if len(right_x) > 100:

        right_fit = np.polyfit(
            right_y,
            right_x,
            2
        )

    plot_y = np.linspace(
        0,
        height - 1,
        height
    )

    left_fit_x = None
    right_fit_x = None

    if left_fit is not None:

        left_fit_x = (
            left_fit[0] * plot_y ** 2
            +
            left_fit[1] * plot_y
            +
            left_fit[2]
        )

    if right_fit is not None:

        right_fit_x = (
            right_fit[0] * plot_y ** 2
            +
            right_fit[1] * plot_y
            +
            right_fit[2]
        )

    return (
        plot_y,
        left_fit_x,
        right_fit_x
    )


def draw_lane(
    original_image,
    birdseye_image,
    matrix,
    plot_y,
    left_fit_x,
    right_fit_x
):
    """
    Draw detected lane curves and transform
    them back to the original perspective.
    """

    height, width = original_image.shape[:2]

    lane_overlay = np.zeros(
    (
        birdseye_image.shape[0],
        birdseye_image.shape[1],
        3
    ),
    dtype=np.uint8
)

    if left_fit_x is not None:

        left_points = np.array(
            [
                np.transpose(
                    np.vstack(
                        [
                            left_fit_x,
                            plot_y
                        ]
                    )
                )
            ],
            dtype=np.int32
        )

        cv2.polylines(
            lane_overlay,
            left_points,
            False,
            (0, 255, 0),
            8
        )

    if right_fit_x is not None:

        right_points = np.array(
            [
                np.transpose(
                    np.vstack(
                        [
                            right_fit_x,
                            plot_y
                        ]
                    )
                )
            ],
            dtype=np.int32
        )

        cv2.polylines(
            lane_overlay,
            right_points,
            False,
            (0, 255, 0),
            8
        )

    inverse_matrix = np.linalg.inv(matrix)

    original_lane = cv2.warpPerspective(
        lane_overlay,
        inverse_matrix,
        (width, height)
    )

    result = cv2.addWeighted(
        original_image,
        1.0,
        original_lane,
        1.0,
        0
    )

    return result


def main():

    if not INPUT_IMAGE.exists():

        raise FileNotFoundError(
            f"Input image not found: {INPUT_IMAGE}"
        )

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:

        raise ValueError(
            "Could not read input image."
        )

    print("Creating white lane mask...")

    white_mask = create_white_mask(
        image
    )
    # Remove sky and upper image region.
    height, width = white_mask.shape

    road_mask = np.zeros_like(
    white_mask
)

    road_mask[
    int(height * 0.50):,
    :
] = 255

    white_mask = cv2.bitwise_and(
    white_mask,
    road_mask
)
    white_mask_path = (
    BASE_DIR
    / "outputs"
    / "images"
    / "white_lane_mask.png"
)

    cv2.imwrite(
    str(white_mask_path),
    white_mask
)

    print(
    f"White mask: {white_mask_path}"
)

    print(
    f"White pixels detected: "
    f"{cv2.countNonZero(white_mask)}"
)
    print("Applying perspective transform...")

    birdseye, matrix = perspective_transform(
        white_mask
    )
    print(
    f"Birdseye white pixels: "
    f"{cv2.countNonZero(birdseye)}"
)
    print("Finding lane pixels...")

    (
        left_x,
        left_y,
        right_x,
        right_y
    ) = find_lane_pixels(
        birdseye
    )

    print(
        f"Left lane pixels : {len(left_x)}"
    )

    print(
        f"Right lane pixels: {len(right_x)}"
    )

    print("Fitting lane curves...")

    (
        plot_y,
        left_fit_x,
        right_fit_x
    ) = fit_lane_polynomial(
        left_x,
        left_y,
        right_x,
        right_y,
        image.shape
    )

    print("Creating final lane result...")

    result = draw_lane(
        image,
        birdseye,
        matrix,
        plot_y,
        left_fit_x,
        right_fit_x
    )

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(BIRDSEYE_IMAGE),
        birdseye
    )

    cv2.imwrite(
        str(OUTPUT_IMAGE),
        result
    )

    print()
    print("Advanced lane detection completed.")
    print(f"Birdseye: {BIRDSEYE_IMAGE}")
    print(f"Output  : {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()