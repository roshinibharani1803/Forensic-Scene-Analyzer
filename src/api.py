from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from pathlib import Path
import tempfile

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Forensic Scene Analyzer API",
    description="YOLOv8-based forensic scene analysis service",
    version="1.0.0"
)

# --------------------------------------------------
# Load Model Once at Startup
# --------------------------------------------------

MODEL_PATH = Path("models/best.pt")

print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Forensic Scene Analyzer API running",
        "status": "healthy"
    }

# --------------------------------------------------
# Model Information Endpoint
# --------------------------------------------------

@app.get("/model-info")
def model_info():
    return {
        "task": model.task,
        "classes": model.names
    }

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    # Run inference
    results = model(temp_path)

    detections = []

    # Extract detected objects
    if results[0].boxes is not None:
        for box in results[0].boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": round(confidence, 4)
                }
            )

    return {
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": detections
    }