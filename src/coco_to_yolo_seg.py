from pathlib import Path
import json
import zipfile
import shutil
import glob


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASETS_DIR = PROJECT_ROOT / "raw_segmentation_datasets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "segmentation"
TEMP_DIR = PROJECT_ROOT / "temp_segmentation"

SPLITS = ["train", "val", "test"]


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "Gun",
    1: "Knife",
    2: "Window",
    3: "FallenBody",
    4: "BloodStain",
}


def get_class_id(filename: str) -> int:

    name = filename.lower()

    if "gun" in name:
        return 0

    if "knife" in name:
        return 1

    if "win" in name:
        return 2

    if "fall" in name or "fallen" in name or "body" in name:
        return 3

    if "blood" in name or "stain" in name or "passive" in name:
        return 4

    raise ValueError(f"Cannot determine class for file: {filename}")


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_structure():

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:

        (OUTPUT_DIR / split / "images").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_DIR / split / "labels").mkdir(
            parents=True,
            exist_ok=True
        )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# ZIP EXTRACTION
# ============================================================

def extract_zip(zip_path: Path):

    # Create a shorter extraction folder name
    safe_name = (
        zip_path.stem
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )[:20]

    extract_dir = TEMP_DIR / safe_name

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    except FileNotFoundError:
        print(
            f"\n❌ Windows path length issue while extracting {zip_path.name}"
        )
        print(
            "Move the project closer to the drive root "
            "(e.g. C:\\FSA) and try again."
        )
        raise

    return extract_dir
# ============================================================
# JSON SEARCH
# ============================================================

def find_annotation_json(extract_dir, split):

    search_terms = [split]

    if split == "val":
        search_terms.append("valid")

    for term in search_terms:

        json_files = glob.glob(
            str(extract_dir / "**" / term / "*annotations.coco.json"),
            recursive=True
        )

        if not json_files:

            json_files = glob.glob(
                str(extract_dir / "**" / term / "annotations.json"),
                recursive=True
            )

        if json_files:
            return Path(json_files[0])

    return None


# ============================================================
# IMAGE SEARCH
# ============================================================

def locate_image(extract_dir, json_path, filename):

    candidate = (
        json_path.parent.parent /
        "images" /
        filename
    )

    if candidate.exists():
        return candidate

    matches = glob.glob(
        str(extract_dir / "**" / filename),
        recursive=True
    )

    if matches:
        return Path(matches[0])

    return None


# ============================================================
# COCO TO YOLO SEGMENTATION
# ============================================================

def process_dataset(zip_path):

    class_id = get_class_id(zip_path.name)

    prefix = (
        zip_path.stem
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    print(f"\nProcessing {zip_path.name}")
    print(f"Class: {CLASS_NAMES[class_id]}")

    extract_dir = extract_zip(zip_path)

    for split in SPLITS:

        json_path = find_annotation_json(
            extract_dir,
            split
        )

        if not json_path:
            print(f"Skipping {split}")
            continue

        print(f"Found {split} annotations")

        with open(json_path, "r") as f:
            coco = json.load(f)

        images = {
            img["id"]: img
            for img in coco["images"]
        }

        image_annotations = {
            img_id: []
            for img_id in images.keys()
        }

        for ann in coco["annotations"]:

            image_id = ann["image_id"]

            if image_id in image_annotations:
                image_annotations[image_id].append(ann)

        image_dst = OUTPUT_DIR / split / "images"
        label_dst = OUTPUT_DIR / split / "labels"

        for image_id, annotations in image_annotations.items():

            image_info = images[image_id]

            width = image_info["width"]
            height = image_info["height"]

            filename = image_info["file_name"]

            image_src = locate_image(
                extract_dir,
                json_path,
                filename
            )

            if image_src is None:
                continue

            base_name = Path(filename).stem
            extension = Path(filename).suffix

            new_name = f"{prefix}_{base_name}"

            shutil.copy2(
                image_src,
                image_dst / f"{new_name}{extension}"
            )

            label_file = (
                label_dst /
                f"{new_name}.txt"
            )

            with open(label_file, "w") as out:

                for ann in annotations:

                    segmentations = ann.get(
                        "segmentation",
                        []
                    )

                    if not segmentations:
                        continue

                    for polygon in segmentations:

                        if len(polygon) < 6:
                            continue

                        normalized = []

                        for i in range(
                            0,
                            len(polygon),
                            2
                        ):

                            x = polygon[i] / width
                            y = polygon[i + 1] / height

                            normalized.append(
                                f"{x:.6f}"
                            )

                            normalized.append(
                                f"{y:.6f}"
                            )

                        line = (
                            str(class_id)
                            + " "
                            + " ".join(normalized)
                            + "\n"
                        )

                        out.write(line)

        print(f"{split} complete")


# ============================================================
# DATA YAML
# ============================================================

def create_data_yaml():

    yaml_file = OUTPUT_DIR / "data.yaml"

    with open(yaml_file, "w") as f:

        f.write(f"path: {OUTPUT_DIR}\n\n")

        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")

        f.write("names:\n")

        for idx, name in CLASS_NAMES.items():
            f.write(f"  {idx}: {name}\n")


# ============================================================
# SUMMARY
# ============================================================

def print_summary():

    print("\n" + "=" * 60)
    print("SEGMENTATION DATASET SUMMARY")
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

    print("\nStarting COCO segmentation merge...")

    create_structure()

    zip_files = sorted(
        RAW_DATASETS_DIR.glob("*.zip")
    )

    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP files found in {RAW_DATASETS_DIR}"
        )

    for zip_file in zip_files:
        process_dataset(zip_file)

    create_data_yaml()

    print_summary()

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    print("\nSegmentation dataset created successfully.")


if __name__ == "__main__":
    main()