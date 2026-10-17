from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "road_damage"
    / "data.yaml"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("ROAD DAMAGE YOLO TRAINING")
    print("=" * 60)

    print(f"Dataset: {DATA_YAML}")

    # Start with a lightweight pretrained model.
    # This is appropriate for your current hardware.
    model = YOLO("yolo11n.pt")

    print()
    print("Starting training...")

    results = model.train(
        data=str(DATA_YAML),
        epochs=15,
        imgsz=416,
        batch=8,
        workers=2,
        project=str(MODEL_DIR),
        name="road_damage_yolo",
        exist_ok=True,
        pretrained=True,
        patience=8,
        cache=False,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Results saved under: {MODEL_DIR}")


if __name__ == "__main__":
    main()