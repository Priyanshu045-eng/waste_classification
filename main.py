from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn


app = FastAPI(
    title="🗑️ AI Waste Classification API",
    description="Classifies waste and returns category if confidence is high enough",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")


CLASS_NAMES = [
    'cardboard', 'glass', 'metal', 'paper', 'plastic',
    'trash', 'organic', 'battery', 'biological', 'clothes', 'shoes',
]

CATEGORY_MAP = {
    'cardboard':  'Recyclable',
    'glass':      'Recyclable',
    'metal':      'Recyclable',
    'paper':      'Recyclable',
    'plastic':    'Recyclable',
    'trash':      'General',
    'organic':    'Biodegradable',
    'battery':    'Hazardous',
    'biological': 'Hazardous',
    'clothes':    'General',
    'shoes':      'General',
}

CONFIDENCE_THRESHOLD = 0.65  


def parse_detections(boxes):
    """Parse YOLO boxes, remove duplicate bounding boxes."""
    if boxes is None or len(boxes) == 0:
        return []

    all_dets = []
    for box in boxes:
        cls_id     = int(box.cls.item())
        confidence = round(float(box.conf.item()), 4)
        cls_name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
        category   = CATEGORY_MAP.get(cls_name, 'General')
        x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
        all_dets.append({
            "class":      cls_name,
            "category":   category,
            "confidence": confidence,
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    all_dets = sorted(all_dets, key=lambda x: x['confidence'], reverse=True)

    unique = []
    seen_boxes = []
    for det in all_dets:
        bbox = (det['bbox']['x1'], det['bbox']['y1'],
                det['bbox']['x2'], det['bbox']['y2'])
        is_dup = any(
            abs(bbox[0]-s[0]) < 50 and abs(bbox[2]-s[2]) < 50 and
            abs(bbox[1]-s[1]) < 50 and abs(bbox[3]-s[3]) < 50
            for s in seen_boxes
        )
        if not is_dup:
            unique.append(det)
            seen_boxes.append(bbox)

    return unique



@app.get("/")
def home():
    return {
        "message":              "🗑️ AI Waste Classification API v4 is running!",
        "docs":                 "/docs",
        "version":              "4.0.0",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "logic":                "Returns waste category only if confidence >= threshold"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH}

@app.get("/classes")
def get_classes():
    return {
        "total_classes": len(CLASS_NAMES),
        "classes": [
            {
                "id":       i,
                "name":     cls,
                "category": CATEGORY_MAP.get(cls, 'General'),
            }
            for i, cls in enumerate(CLASS_NAMES)
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Only image files accepted!"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(image, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False)[0]
        detections = parse_detections(results.boxes)

        confident_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]

        if not confident_detections:
            return JSONResponse(content={
                "status":   "low_confidence",
                "detected": False,
                "message":  "📸 Please provide a clearer image for accurate waste detection.",
                "tip":      "Ensure the waste item is well-lit, in focus, and centered in the frame."
            })

        top = confident_detections[0]

        return JSONResponse(content={
            "status":     "success",
            "detected":   True,
            "class":      top['class'],
            "category":   top['category'],
            "confidence": top['confidence'],
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    results_list = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(image, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False)[0]
        detections = parse_detections(results.boxes)
        confident_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]

        if confident_detections:
            top = confident_detections[0]
            results_list.append({
                "filename":   file.filename,
                "detected":   True,
                "class":      top['class'],
                "category":   top['category'],
                "confidence": top['confidence'],
            })
        else:
            results_list.append({
                "filename": file.filename,
                "detected": False,
                "status":   "low_confidence",
                "message":  "📸 Please provide a clearer image for accurate waste detection.",
                "tip":      "Ensure the waste item is well-lit, in focus, and centered in the frame."
            })

    return JSONResponse(content={
        "status":       "success",
        "total_images": len(files),
        "results":      results_list
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
