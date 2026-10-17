from pathlib import Path
import cv2

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "road_damage_yolo"
    / "weights"
    / "best.pt"
)

IMAGE_PATH = r"data\processed\road_damage\images\val\India_000010.jpg"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "images"
    / "road_damage_prediction"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("ROAD DAMAGE PREDICTION")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"Image : {IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.40,
    imgsz=416,
    verbose=True,
)

    result = results[0]

# Create image with bounding boxes and labels
    annotated_image = result.plot()

# Save annotated image explicitly
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "road_damage_result.jpg"
    cv2.imwrite(str(output_file), annotated_image)

    print(f"Annotated result saved to: {output_file}")

    result = results[0]

    print()
    print("Detections:")

    if result.boxes is None or len(result.boxes) == 0:
        print("No road damage detected.")
    else:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                f" - {class_name}: "
                f"{confidence:.2f}"
            )

    print()
    print("Prediction completed.")
    print(
        f"Output: "
        f"{OUTPUT_DIR / 'result'}"
    )


if __name__ == "__main__":
    main()