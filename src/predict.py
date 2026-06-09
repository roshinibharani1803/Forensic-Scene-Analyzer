from pathlib import Path
import argparse

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Forensic Scene Analyzer Inference"
    )

    parser.add_argument(
        "image",
        help="Path to input image"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )

    return parser.parse_args()


def predict():

    args = parse_arguments()

    image_path = Path(args.image)
    confidence = args.conf

    if not image_path.exists():
        print(f"\nError: Image not found -> {image_path}")
        return

    print("\nLoading model...")

    model = YOLO(MODEL_PATH)

    print(f"Image: {image_path}")
    print(f"Confidence Threshold: {confidence}")

    print("\nRunning inference...")

    results = model.predict(
        source=str(image_path),
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
        conf=confidence
    )

    print("\nPrediction completed.")
    print(f"Results saved to: {OUTPUT_DIR / 'predictions'}")

    return results


if __name__ == "__main__":
    predict()