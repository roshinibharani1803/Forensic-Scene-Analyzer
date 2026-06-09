from pathlib import Path
import yaml
from ultralytics import YOLO


def load_config():

    config_path = (
        Path(__file__)
        .resolve()
        .parent.parent
        / "configs"
        / "segmentation.yaml"
    )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train():

    config = load_config()

    print("\nLoading model...")
    model = YOLO(config["model"])

    print("Starting training...\n")

    model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        project=config["project"],
        name=config["name"],
        device=config["device"]
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train()