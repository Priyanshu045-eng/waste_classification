from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn


app = FastAPI(
    title="🗑️ AI Waste Classification API",
    description="Classifies waste — always returns the most harmful category first",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load Model
# ============================================================
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

# Higher number = more harmful = gets priority
HARM_PRIORITY = {
    'Hazardous':    4,
    'General':      3,
    'Biodegradable':2,
    'Recyclable':   1,
}

HARM_EMOJI = {
    'Hazardous':    '🔴',
    'General':      '🔘',
    'Biodegradable':'🟢',
    'Recyclable':   '🔵',
}

HARM_MESSAGE = {
    'Hazardous':     '⚠️ HAZARDOUS — Handle with care! Dispose at designated hazardous waste facility.',
    'General':       '🔘 GENERAL WASTE — Non-recyclable. Dispose in general waste bin.',
    'Biodegradable': '🟢 BIODEGRADABLE — Can be composted. Dispose in organic/green bin.',
    'Recyclable':    '♻️ RECYCLABLE — Clean and place in recycling bin.',
}

INCENTIVE_POINTS = {
    'Recyclable':    10,
    'Biodegradable': 8,
    'Hazardous':     5,
    'General':       1,
}

# ============================================================
# Helper Functions
# ============================================================

def get_most_harmful(detections):
    """From all detections, return the most harmful one.
    If tie in harm priority, return highest confidence."""
    if not detections:
        return None
    return sorted(
        detections,
        key=lambda x: (HARM_PRIORITY.get(x['category'], 0), x['confidence']),
        reverse=True
    )[0]


def calculate_score(detections):
    if not detections:
        return 0, 'N/A'
    total = sum(INCENTIVE_POINTS.get(d['category'], 0) * d['confidence'] for d in detections)
    max_pts = len(detections) * 10
    score = round(min(100, (total / max_pts) * 100), 1)
    if score >= 80:   grade = 'A'
    elif score >= 60: grade = 'B'
    elif score >= 40: grade = 'C'
    else:             grade = 'D'
    return score, grade


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
            "class":         cls_name,
            "category":      category,
            "confidence":    confidence,
            "harm_priority": HARM_PRIORITY.get(category, 0),
            "bbox":          {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    # Sort by confidence
    all_dets = sorted(all_dets, key=lambda x: x['confidence'], reverse=True)

    # Remove duplicate bounding boxes
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


# ============================================================
# Routes
# ============================================================

@app.get("/")
def home():
    return {
        "message":        "🗑️ AI Waste Classification API v3 is running!",
        "docs":           "/docs",
        "version":        "3.0.0",
        "priority_logic": "Most harmful category always wins"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH}

@app.get("/classes")
def get_classes():
    return {
        "total_classes": len(CLASS_NAMES),
        "harm_priority_order": [
            "🔴 Hazardous     (Priority 4 — Highest)",
            "🔘 General       (Priority 3)",
            "🟢 Biodegradable (Priority 2)",
            "🔵 Recyclable    (Priority 1 — Lowest)",
        ],
        "classes": [
            {
                "id":            i,
                "name":          cls,
                "category":      CATEGORY_MAP.get(cls, 'General'),
                "harm_priority": HARM_PRIORITY.get(CATEGORY_MAP.get(cls, 'General'), 0)
            }
            for i, cls in enumerate(CLASS_NAMES)
        ]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a waste image.
    If multiple categories detected → MOST HARMFUL wins.
    Priority: Hazardous > General > Biodegradable > Recyclable
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Only image files accepted!"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Try multiple confidence thresholds
        detections = []
        used_conf = 0.25
        for conf_threshold in [0.45, 0.35, 0.25, 0.15]:
            results = model(image, conf=conf_threshold, iou=0.45, verbose=False)[0]
            detections = parse_detections(results.boxes)
            if detections:
                used_conf = conf_threshold
                break

        # ── CORE LOGIC: Pick most harmful ──
        most_harmful = get_most_harmful(detections)
        # ───────────────────────────────────

        score, grade = calculate_score(detections)

        category_summary = {}
        for d in detections:
            category_summary[d['category']] = category_summary.get(d['category'], 0) + 1

        final_result = None
        if most_harmful:
            final_result = {
                "class":         most_harmful['class'],
                "category":      most_harmful['category'],
                "confidence":    most_harmful['confidence'],
                "harm_priority": most_harmful['harm_priority'],
                "emoji":         HARM_EMOJI.get(most_harmful['category'], ''),
                "action":        HARM_MESSAGE.get(most_harmful['category'], ''),
            }

        return JSONResponse(content={
            "status":           "success",
            "filename":         file.filename,
            "total_detections": len(detections),
            "final_result":     final_result,        # ✅ Most harmful
            "all_detections":   detections,           # All for reference
            "category_summary": category_summary,
            "incentive_score":  score,
            "grade":            grade,
            "confidence_used":  used_conf,
            "message": (
                f"⚠️ Most harmful: {most_harmful['category']} ({most_harmful['class']})"
                if most_harmful else "No waste detected"
            )
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch classify — each image returns its most harmful category."""
    results_list = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        detections = []
        for conf_threshold in [0.45, 0.35, 0.25, 0.15]:
            results = model(image, conf=conf_threshold, verbose=False)[0]
            detections = parse_detections(results.boxes)
            if detections:
                break

        most_harmful = get_most_harmful(detections)
        score, grade = calculate_score(detections)

        results_list.append({
            "filename": file.filename,
            "final_result": {
                "class":      most_harmful['class']      if most_harmful else None,
                "category":   most_harmful['category']   if most_harmful else None,
                "confidence": most_harmful['confidence'] if most_harmful else None,
                "emoji":      HARM_EMOJI.get(most_harmful['category'], '') if most_harmful else None,
                "action":     HARM_MESSAGE.get(most_harmful['category'], '') if most_harmful else None,
            },
            "total_detections": len(detections),
            "incentive_score":  score,
            "grade":            grade,
        })

    return JSONResponse(content={
        "status":       "success",
        "total_images": len(files),
        "results":      results_list
    })


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
