from pathlib import Path
import yaml
from ultralytics import YOLO


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():

    project_root = Path(__file__).resolve().parent.parent

    config_path = project_root / "configs" / "detection.yaml"

    config = load_config(config_path)

    print("=" * 60)
    print("FORENSIC SCENE ANALYZER - DETECTION TRAINING")
    print("=" * 60)

    print(f"Model      : {config['model']}")
    print(f"Dataset    : {config['data']}")
    print(f"Epochs     : {config['epochs']}")
    print(f"Image Size : {config['imgsz']}")
    print(f"Batch Size : {config['batch']}")

    model = YOLO(config["model"])

    results = model.train(
        data=str(project_root / config["data"]),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        project=str(project_root / "outputs"),
        name=config["name"],
        device=config["device"]
    )

    print("\nTraining completed successfully.")
    print(f"Results saved in outputs/{config['name']}")

    return results


if __name__ == "__main__":
    main()