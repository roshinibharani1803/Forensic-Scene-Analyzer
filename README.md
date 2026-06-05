# Forensic Scene Analyzer

AI-powered forensic evidence localization using YOLOv8 Detection and Segmentation.

## Overview

Forensic investigations often require the identification and localization of critical evidence from crime scene images. This project utilizes state-of-the-art computer vision techniques to automatically detect and segment forensic evidence such as firearms, knives, blood stains, broken windows, and fallen bodies.

The system combines multiple datasets, performs dataset harmonization, and trains YOLOv8-based object detection and instance segmentation models to assist forensic scene analysis.

## Features

* Multi-class forensic evidence detection
* Instance segmentation of crime-scene objects
* Automated dataset merging pipeline
* COCO-to-YOLO segmentation conversion
* YOLOv8 Detection and Segmentation training
* Modular and scalable project structure
* Deployment-ready architecture

## Target Classes

| Class ID | Class Name  |
| -------- | ----------- |
| 0        | Gun         |
| 1        | Knife       |
| 2        | Window      |
| 3        | Fallen Body |
| 4        | Blood Stain |

## Dataset Statistics

### Detection Dataset

| Split      | Images |
| ---------- | ------ |
| Train      | 1537   |
| Validation | 341    |
| Test       | 182    |

Total Images: 2060

The dataset was created by merging multiple YOLO-format datasets and mapping all labels into a unified class space.

## Project Structure

```text
Forensic-Scene-Analyzer/

├── assets/
├── configs/
│   ├── detection.yaml
│   └── segmentation.yaml
│
├── data/
│   ├── detection/
│   └── segmentation/
│
├── models/
├── notebooks/
│   └── training_pipeline.ipynb
│
├── outputs/
├── raw_datasets/
├── raw_segmentation_datasets/
├── samples/
│
├── src/
│   ├── merge_yolo_datasets.py
│   ├── coco_to_yolo_seg.py
│   ├── train_detection.py
│   ├── train_segmentation.py
│   └── inference.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Clone Repository

```bash
git clone https://github.com/roshinibharani1803/Forensic-Scene-Analyzer.git
cd Forensic-Scene-Analyzer
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Place the YOLO-format datasets inside:

```text
raw_datasets/
```

Run:

```bash
python src/merge_yolo_datasets.py
```

The script automatically:

* Extracts datasets
* Harmonizes class labels
* Merges train, validation, and test splits
* Generates a YOLO-compatible configuration file

## Training

### Detection Model

Configure training parameters in:

```text
configs/detection.yaml
```

Run:

```bash
python src/train_detection.py
```

### Segmentation Model

Configure training parameters in:

```text
configs/segmentation.yaml
```

Run:

```bash
python src/train_segmentation.py
```

## Inference

Run:

```bash
python src/inference.py
```

Generated outputs are stored in:

```text
outputs/
```

## Technologies Used

* Python
* YOLOv8
* Ultralytics
* PyTorch
* OpenCV
* NumPy
* Gradio
* Hugging Face Spaces

## Future Enhancements

* Real-time video analysis
* Multi-modal forensic scene understanding
* Evidence relationship mapping
* 3D crime scene reconstruction
* Cloud-native deployment pipeline

## Author

Roshini Bharani

Bachelor of Engineering in Artificial Intelligence and Machine Learning

BMS College of Engineering
