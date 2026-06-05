from pathlib import Path
import zipfile
import shutil
import glob

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASETS_DIR = PROJECT_ROOT / "raw_datasets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "detection"
TEMP_DIR = PROJECT_ROOT / "temp_detection"

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "Gun",
    1: "Knife",
    2: "Window",
    3: "FallenBody",
    4: "BloodStain",
}


# ============================================================
# CLASS DETECTION
# ============================================================

def get_class_id(zip_name: str) -> int:
    name = zip_name.lower()

    if "gun" in name:
        return 0

    if "knife" in name:
        return 1

    if "window" in name or "win" in name:
        return 2

    if "fall" in name or "fallen" in name or "body" in name:
        return 3

    if "blood" in name or "stain" in name:
        return 4

    raise ValueError(f"Unable to determine class from filename: {zip_name}")


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directory_structure():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ZIP EXTRACTION
# ============================================================

def extract_dataset(zip_path: Path) -> Path:
    extract_folder = TEMP_DIR / zip_path.stem

    if extract_folder.exists():
        shutil.rmtree(extract_folder)

    extract_folder.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    return extract_folder


# ============================================================
# DATASET MERGING
# ============================================================

def merge_dataset(zip_path: Path):
    class_id = get_class_id(zip_path.name)

    prefix = (
        zip_path.stem
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    print(f"\nProcessing: {zip_path.name}")
    print(f"Class: {CLASS_NAMES[class_id]}")

    extracted_dir = extract_dataset(zip_path)

    for split in SPLITS:

        image_dirs = glob.glob(
            str(extracted_dir / "**" / split / "images"),
            recursive=True
        )

        if not image_dirs:
            print(f"  Skipping {split} split")
            continue

        for image_dir in image_dirs:

            label_dir = image_dir.replace("images", "labels")

            if not Path(label_dir).exists():
                continue

            image_output_dir = OUTPUT_DIR / split / "images"
            label_output_dir = OUTPUT_DIR / split / "labels"

            image_files = glob.glob(f"{image_dir}/*")

            for image_path in image_files:

                image_path = Path(image_path)

                base_name = image_path.stem
                extension = image_path.suffix

                new_base_name = f"{prefix}_{base_name}"

                destination_image = (
                    image_output_dir /
                    f"{new_base_name}{extension}"
                )

                destination_label = (
                    label_output_dir /
                    f"{new_base_name}.txt"
                )

                shutil.copy2(image_path, destination_image)

                original_label = (
                    Path(label_dir) /
                    f"{base_name}.txt"
                )

                if original_label.exists():

                    with open(original_label, "r") as fin:
                        lines = fin.readlines()

                    with open(destination_label, "w") as fout:

                        for line in lines:

                            parts = line.strip().split()

                            if not parts:
                                continue

                            parts[0] = str(class_id)

                            fout.write(
                                " ".join(parts) + "\n"
                            )

    print("Completed")


# ============================================================
# DATA YAML CREATION
# ============================================================

def create_data_yaml():
    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_text = f"""path: {OUTPUT_DIR}

train: train/images
val: valid/images
test: test/images

names:
"""

    for class_id, class_name in CLASS_NAMES.items():
        yaml_text += f"  {class_id}: {class_name}\n"

    with open(yaml_path, "w") as f:
        f.write(yaml_text)

    print(f"\ndata.yaml created: {yaml_path}")


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary():
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for split in SPLITS:

        images = list(
            (OUTPUT_DIR / split / "images").glob("*")
        )

        labels = list(
            (OUTPUT_DIR / split / "labels").glob("*.txt")
        )

        print(
            f"{split:<8} | "
            f"Images: {len(images):<5} | "
            f"Labels: {len(labels):<5}"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nStarting YOLO dataset merge...")

    create_directory_structure()

    zip_files = sorted(
        RAW_DATASETS_DIR.glob("*.zip")
    )

    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP files found in {RAW_DATASETS_DIR}"
        )

    for zip_file in zip_files:
        merge_dataset(zip_file)

    create_data_yaml()
    print_summary()

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    print("\nDataset merge completed successfully.")


if __name__ == "__main__":
    main()