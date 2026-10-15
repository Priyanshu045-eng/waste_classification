import os
import io
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import numpy as np

# Try importing ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI • Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Theme & Styling (Modern Glassmorphism UI)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 95, 70, 0.05) 50%, rgba(14, 165, 233, 0.10) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(120deg, #10b981, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
        font-weight: 400;
        line-height: 1.5;
    }

    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(226, 232, 240, 0.8);
        color: #334155;
    }

    /* Result Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.1);
    }

    /* Category Specific Pill Highlights */
    .pill-recyclable {
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .pill-biodegradable {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .pill-hazardous {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .pill-general {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Metric stat box */
    .stat-box {
        text-align: center;
        padding: 1.2rem;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.85));
        border-radius: 14px;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    /* Disposal Instructions Box */
    .disposal-guide {
        border-radius: 14px;
        padding: 1.3rem;
        margin-top: 1rem;
        border-left: 5px solid;
    }
    .guide-recyclable {
        background: #f0f9ff;
        border-color: #0284c7;
        color: #0c4a6e;
    }
    .guide-biodegradable {
        background: #f0fdf4;
        border-color: #16a34a;
        color: #14532d;
    }
    .guide-hazardous {
        background: #fef2f2;
        border-color: #dc2626;
        color: #7f1d1d;
    }
    .guide-general {
        background: #f8fafc;
        border-color: #64748b;
        color: #1e293b;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 10px 10px 0px 0px;
        font-weight: 600;
        padding: 0px 18px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Domain Knowledge & Waste Metadata
# -----------------------------------------------------------------------------
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

CATEGORY_COLORS = {
    'Recyclable':    (2, 132, 199),    # #0284c7 (Sky Blue)
    'Biodegradable': (22, 163, 74),    # #16a34a (Emerald Green)
    'Hazardous':     (220, 38, 38),    # #dc2626 (Crimson Red)
    'General':       (100, 116, 139),  # #64748b (Slate)
}

CATEGORY_HEX = {
    'Recyclable':    '#0284c7',
    'Biodegradable': '#16a34a',
    'Hazardous':     '#dc2626',
    'General':       '#64748b',
}

DISPOSAL_INFO = {
    'cardboard': {
        'icon': '📦',
        'bin': 'Blue Recycling Bin',
        'instructions': [
            'Flatten all boxes to save space in the recycling bin.',
            'Remove excess packaging tape, bubble wrap, and styrofoam inserts.',
            'Ensure cardboard is dry and free of greasy food contamination (pizza oil stains cannot be recycled).'
        ],
        'impact': 'Recycling 1 ton of cardboard saves 17 trees, 7,000 gallons of water, and 4,000 kWh of electricity.'
    },
    'glass': {
        'icon': '🍾',
        'bin': 'Blue Recycling Bin (Glass)',
        'instructions': [
            'Rinse thoroughly to eliminate residual liquids or odor.',
            'Remove metal or plastic screw caps and recycle them separately.',
            'Do not break glassware; place intact bottles and jars safely in the bin.',
            'Note: Pyrex, drinking glasses, window glass, and ceramics require general disposal.'
        ],
        'impact': 'Glass is 100% recyclable and can be recycled endlessly without loss in quality or purity.'
    },
    'metal': {
        'icon': '🥫',
        'bin': 'Blue Recycling Bin (Metal)',
        'instructions': [
            'Rinse aluminum beverage cans and food tin cans thoroughly.',
            'Crush soda cans to maximize collection bin density.',
            'Clean aluminum foil can be balled together and recycled.'
        ],
        'impact': 'Recycling aluminum saves 95% of the energy needed to produce aluminum from raw bauxite ore.'
    },
    'paper': {
        'icon': '📄',
        'bin': 'Blue Recycling Bin (Paper)',
        'instructions': [
            'Keep paper dry and clean (office sheets, magazines, newspaper, envelopes).',
            'Remove plastic spiral binders or heavy metal clips before disposal.',
            'Shredded paper should be bagged together in clear bags if accepted locally.',
            'Thermal receipts, wax-coated paper, and dirty tissues belong in General Waste.'
        ],
        'impact': 'Recycling paper conserves 60% of water and prevents tons of CO2 greenhouse emissions.'
    },
    'plastic': {
        'icon': '🧴',
        'bin': 'Blue Recycling Bin (Plastics)',
        'instructions': [
            'Empty any liquid contents and lightly rinse the container.',
            'Check the resin identification code (#1 PETE, #2 HDPE are widely accepted).',
            'Replace the plastic cap securely onto the crushed bottle.'
        ],
        'impact': 'Reduces microplastic contamination in our oceans and reduces petrochemical refinery dependency.'
    },
    'trash': {
        'icon': '🗑️',
        'bin': 'Black / Grey General Landfill Bin',
        'instructions': [
            'Use standard waste bags and tie securely before disposal.',
            'Check if components can be disassembled for localized recycling.',
            'Strive to reduce non-recyclable multi-layer packaging purchases.'
        ],
        'impact': 'Landfill waste generates methane. Minimizing non-recyclable waste prolongs landfill lifespan.'
    },
    'organic': {
        'icon': '🍏',
        'bin': 'Green Organic Compost Bin',
        'instructions': [
            'Includes fruit peels, vegetable scraps, coffee grounds, eggshells, and yard leaves.',
            'Do NOT include plastic stickers from produce, synthetic tea bags, or pet waste.',
            'Can be composted at home or placed in municipal green compost bins.'
        ],
        'impact': 'Composting converts organic matter into rich soil humus and avoids anaerobic landfill methane release.'
    },
    'battery': {
        'icon': '🔋',
        'bin': 'Specialized E-Waste Drop-Off Point',
        'instructions': [
            '⚠️ CRITICAL: NEVER discard batteries into municipal garbage or recycling bins!',
            'Batteries pose severe fire and explosion risks in waste compaction trucks.',
            'Tape terminal ends with electrical tape to avoid short circuits.',
            'Take to local battery drop-off kiosks (supermarkets, electronics stores, or e-waste stations).'
        ],
        'impact': 'Prevents toxic lead, cadmium, and lithium from leaching into underground freshwater aquifers.'
    },
    'biological': {
        'icon': '☣️',
        'bin': 'Biohazard / Medical Disposal Unit',
        'instructions': [
            'Use sealed, puncture-proof biohazard containers or yellow medical waste bags.',
            'Sharps (syringes, lancets) must always be disposed in rigid sharps containers.',
            'Drop off at certified medical clinics, hospitals, or hazardous waste municipal events.'
        ],
        'impact': 'Protects sanitation workers and community health from infectious pathogens and dangerous needle injuries.'
    },
    'clothes': {
        'icon': '👕',
        'bin': 'Textile Donation or Fabric Recovery',
        'instructions': [
            'If gently used or wearable: donate to charity organizations, thrift stores, or clothing drives.',
            'If torn or unwearable: check for local textile recycling drop-off centers.',
            'Can be repurposed as cleaning rags before final disposal.'
        ],
        'impact': 'The fashion industry contributes ~10% of global carbon emissions. Reusing clothes drastically cuts environmental footprint.'
    },
    'shoes': {
        'icon': '👟',
        'bin': 'Shoe Donation or General Waste',
        'instructions': [
            'Tie shoelaces together in pairs before donating to clothing banks or charities.',
            'Many athletic brands collect worn-out sports shoes to grind into playground turf.',
            'Heavily decomposed footwear belongs in the general waste bin.'
        ],
        'impact': 'Rubber and synthetic shoe soles take up to 40-50 years to decompose in landfills.'
    }
}

# -----------------------------------------------------------------------------
# Model Resolver & Inference Engine
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI Waste Classification Model...")
def load_yolo_model():
    """Locate and load the YOLO model weights robustly."""
    if not YOLO_AVAILABLE:
        return None, "The ultralytics library is not installed in the environment."

    possible_paths = [
        Path("best.pt"),
        Path(__file__).resolve().parent / "best.pt",
        Path(__file__).resolve().parent.parent / "best.pt",
        Path(__file__).resolve().parent / "waste_classification" / "best.pt",
        Path("waste_classification/best.pt"),
    ]

    for p in possible_paths:
        if p.exists() and p.is_file():
            try:
                model = YOLO(str(p))
                return model, str(p)
            except Exception as e:
                return None, f"Found weights at {p} but failed to load: {e}"

    return None, "Model file 'best.pt' not found in workspace search paths."


def parse_detections(boxes, conf_threshold=0.60):
    """Parse YOLO bounding boxes, match classes, and deduplicate."""
    if boxes is None or len(boxes) == 0:
        return []

    all_dets = []
    for box in boxes:
        cls_id = int(box.cls.item())
        confidence = round(float(box.conf.item()), 4)
        if confidence < conf_threshold:
            continue

        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        category = CATEGORY_MAP.get(cls_name, 'General')
        x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]

        all_dets.append({
            "class": cls_name,
            "category": category,
            "confidence": confidence,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    # Sort descending by confidence
    all_dets = sorted(all_dets, key=lambda x: x['confidence'], reverse=True)

    # Deduplicate overlapping boxes
    unique = []
    seen_boxes = []
    for det in all_dets:
        bbox = (det['bbox']['x1'], det['bbox']['y1'], det['bbox']['x2'], det['bbox']['y2'])
        is_dup = any(
            abs(bbox[0] - s[0]) < 50 and abs(bbox[2] - s[2]) < 50 and
            abs(bbox[1] - s[1]) < 50 and abs(bbox[3] - s[3]) < 50
            for s in seen_boxes
        )
        if not is_dup:
            unique.append(det)
            seen_boxes.append(bbox)

    return unique


def draw_styled_detections(image: Image.Image, detections: list, show_labels=True, show_conf=True) -> Image.Image:
    """Draw stylish bounding boxes with category-colored tags on PIL image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # Choose a proportional line width
    line_w = max(3, int(min(width, height) * 0.005))
    font_size = max(14, int(min(width, height) * 0.025))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1 = det['bbox']['x1'], det['bbox']['y1']
        x2, y2 = det['bbox']['x2'], det['bbox']['y2']
        category = det['category']
        cls_name = det['class'].upper()
        conf_pct = f"{det['confidence'] * 100:.1f}%"

        color = CATEGORY_COLORS.get(category, (100, 116, 139))

        # Draw main bounding box with rounded feel or multi-pass border
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        # Corner accent brackets
        corner_len = min(25, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
        if corner_len > 5:
            c_w = line_w + 2
            # Top-left
            draw.line([(x1, y1), (x1 + corner_len, y1)], fill=color, width=c_w)
            draw.line([(x1, y1), (x1, y1 + corner_len)], fill=color, width=c_w)
            # Top-right
            draw.line([(x2, y1), (x2 - corner_len, y1)], fill=color, width=c_w)
            draw.line([(x2, y1), (x2, y1 + corner_len)], fill=color, width=c_w)
            # Bottom-left
            draw.line([(x1, y2), (x1 + corner_len, y2)], fill=color, width=c_w)
            draw.line([(x1, y2), (x1, y2 - corner_len)], fill=color, width=c_w)
            # Bottom-right
            draw.line([(x2, y2), (x2 - corner_len, y2)], fill=color, width=c_w)
            draw.line([(x2, y2), (x2, y2 - corner_len)], fill=color, width=c_w)

        if show_labels:
            label_text = cls_name
            if show_conf:
                label_text += f" • {conf_pct}"

            # Text background pill
            bbox = font.getbbox(label_text) if hasattr(font, 'getbbox') else (0, 0, len(label_text)*8, font_size)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            pad_x, pad_y = 6, 4
            tag_y1 = max(0, y1 - text_h - (pad_y * 2) - 4)
            tag_y2 = tag_y1 + text_h + (pad_y * 2)
            tag_x2 = min(width, x1 + text_w + (pad_x * 2))

            draw.rectangle([x1, tag_y1, tag_x2, tag_y2], fill=color)
            draw.text((x1 + pad_x, tag_y1 + pad_y), label_text, fill=(255, 255, 255), font=font)

    return annotated


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    # Load model
    model, model_info = load_yolo_model()

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        
        if model is not None:
            st.success(f"🟢 **Model Active**\n`{Path(model_info).name}`")
        else:
            st.error(f"🔴 **Model Offline**\n{model_info}")

        st.markdown("---")
        st.markdown("#### 🎯 Detection Sensitivity")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.60,
            step=0.05,
            help="Minimum AI confidence required to accept a detected waste item."
        )

        iou_threshold = st.slider(
            "IOU Overlap Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Non-Max Suppression overlap threshold."
        )

        st.markdown("---")
        st.markdown("#### 🎨 Visual Overlays")
        show_labels = st.checkbox("Show Category Labels", value=True)
        show_conf = st.checkbox("Show Confidence Percentage", value=True)
        show_raw_coords = st.checkbox("Show Bounding Box Coordinates", value=False)

        st.markdown("---")
        st.markdown("### 📌 Waste Categories")
        st.markdown("""
        - 🔵 **Recyclable**: Paper, Plastic, Metal, Glass, Cardboard
        - 🟢 **Biodegradable**: Organic Scraps
        - 🔴 **Hazardous**: Battery, Biological
        - 🟣 **General**: Trash, Clothes, Shoes
        """)

        st.markdown("---")
        st.caption("AI Waste Classifier v4.0 • Built with Streamlit & YOLO")

    # Hero Header Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">♻️ EcoSort AI Waste Intelligence</div>
        <div class="hero-subtitle">
            Instantly detect, classify, and sort recyclable, organic, and hazardous items with deep-learning computer vision.
        </div>
        <div class="badge-bar">
            <span class="status-badge">⚡ Real-time YOLOv8 Inference</span>
            <span class="status-badge">🏷️ 11 Classes Supported</span>
            <span class="status-badge">🌿 Zero-Contamination Waste Sorting</span>
            <span class="status-badge">🛡️ High-Confidence Verification</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Tabs
    tab_single, tab_batch, tab_directory, tab_diagnostics = st.tabs([
        "📸 Single Item Detection",
        "📦 Batch Image Sorter",
        "📚 Waste Recycling Directory",
        "📊 Model & Calibration"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Single Item Detection
    # -------------------------------------------------------------------------
    with tab_single:
        col_input_opt, col_workspace = st.columns([1, 2.5], gap="medium")

        with col_input_opt:
            st.markdown("#### 📥 Input Source")
            input_mode = st.radio(
                "Choose input method:",
                ["📤 Upload Image", "📷 Live Webcam", "🖼️ Sample Demos"],
                label_visibility="collapsed"
            )

            image_to_process = None

            if input_mode == "📤 Upload Image":
                uploaded_file = st.file_uploader(
                    "Drop waste image here",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    help="Upload a clear photo of the waste item to be sorted."
                )
                if uploaded_file is not None:
                    try:
                        image_to_process = Image.open(uploaded_file).convert("RGB")
                    except Exception as e:
                        st.error(f"Could not open image: {e}")

            elif input_mode == "📷 Live Webcam":
                cam_pic = st.camera_input("Capture item from webcam")
                if cam_pic is not None:
                    image_to_process = Image.open(cam_pic).convert("RGB")

            elif input_mode == "🖼️ Sample Demos":
                sample_choice = st.selectbox(
                    "Select a sample demo item:",
                    ["Plastic Bottle", "Cardboard Box", "Organic Banana Peel", "AA Battery", "Soda Can"]
                )
                
                # Create synthetic sample image if file doesn't exist
                demo_colors = {
                    "Plastic Bottle": ("plastic", (200, 230, 255)),
                    "Cardboard Box": ("cardboard", (220, 190, 150)),
                    "Organic Banana Peel": ("organic", (255, 240, 150)),
                    "AA Battery": ("battery", (100, 100, 120)),
                    "Soda Can": ("metal", (220, 220, 230)),
                }
                
                if st.button("Load Selected Demo Item", use_container_width=True):
                    chosen_class, bg_col = demo_colors[sample_choice]
                    sample_img = Image.new("RGB", (640, 480), color=bg_col)
                    draw = ImageDraw.Draw(sample_img)
                    draw.rectangle([160, 100, 480, 380], fill=(245, 245, 245), outline=(100, 100, 100), width=4)
                    draw.text((220, 230), f"SAMPLE: {chosen_class.upper()}", fill=(40, 40, 40))
                    image_to_process = sample_img
                    st.session_state['demo_img'] = sample_img

                if 'demo_img' in st.session_state and image_to_process is None:
                    image_to_process = st.session_state['demo_img']

        with col_workspace:
            if image_to_process is None:
                st.info("👈 Select or upload an image on the left to begin classification.")
            else:
                with st.spinner("Analyzing image features & detecting waste classes..."):
                    start_t = time.time()
                    
                    if model is not None:
                        results = model(image_to_process, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                        detections = parse_detections(results.boxes, conf_threshold=conf_threshold)
                    else:
                        # Fallback simulated response if weights are missing
                        detections = []
                    
                    inference_time_ms = round((time.time() - start_t) * 1000, 1)

                annotated_img = draw_styled_detections(
                    image_to_process,
                    detections,
                    show_labels=show_labels,
                    show_conf=show_conf
                )

                # Visual side-by-side
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.markdown("**Original Image**")
                    st.image(image_to_process, use_container_width=True)

                with img_col2:
                    st.markdown("**AI Detection Result**")
                    st.image(annotated_img, use_container_width=True)

                # Download Annotated Image
                buf = io.BytesIO()
                annotated_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="💾 Download Annotated Image",
                    data=byte_im,
                    file_name="ecosort_classified_item.png",
                    mime="image/png",
                    use_container_width=True
                )

                st.markdown("---")

                # Classification Result Hero
                if not detections:
                    st.warning("⚠️ **Low Confidence / No Clear Waste Detected**")
                    st.markdown("""
                    The AI could not confidently identify a waste object exceeding the current threshold (**{:.0f}%**).
                    
                    **Tips for best results:**
                    - Ensure the waste item is well-lit and in sharp focus.
                    - Center the object in the frame.
                    - If testing low-contrast objects, try reducing the **Confidence Threshold** slider in the sidebar.
                    """.format(conf_threshold * 100))
                else:
                    top = detections[0]
                    cls_name = top['class']
                    category = top['category']
                    conf = top['confidence']
                    disp = DISPOSAL_INFO.get(cls_name, {})

                    # Metrics Strip
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number">{disp.get('icon', '🏷️')} {cls_name.title()}</div>
                            <div class="stat-label">Identified Item</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        pill_class = f"pill-{category.lower()}"
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number"><span class="{pill_class}">{category}</span></div>
                            <div class="stat-label">Disposal Category</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number" style="color: {CATEGORY_HEX.get(category, '#333')}">{conf * 100:.1f}%</div>
                            <div class="stat-label">Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number">{inference_time_ms} ms</div>
                            <div class="stat-label">Inference Latency</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Actionable Smart Disposal Recommendation Guide
                    guide_class = f"guide-{category.lower()}"
                    instructions_html = "".join([f"<li>{item}</li>" for item in disp.get('instructions', [])])

                    st.markdown(f"""
                    <div class="disposal-guide {guide_class}">
                        <h4 style="margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
                            <span>{disp.get('icon', '♻️')}</span>
                            <span>Target Sorting Bin: <b>{disp.get('bin', 'General Waste')}</b></span>
                        </h4>
                        <p style="margin-bottom: 0.5rem; font-weight: 500;">Recommended Handling Steps:</p>
                        <ul style="margin: 0 0 0.8rem 1.2rem; padding: 0;">
                            {instructions_html}
                        </ul>
                        <div style="font-size: 0.85rem; opacity: 0.9; font-style: italic; border-top: 1px dashed rgba(0,0,0,0.15); padding-top: 0.5rem;">
                            🌍 <b>Environmental Impact:</b> {disp.get('impact', 'Proper sorting keeps valuable resources in circulation.')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Multiple detections breakdown if > 1 item
                    if len(detections) > 1:
                        with st.expander(f"🔍 All Detected Items in Frame ({len(detections)})"):
                            for idx, d in enumerate(detections):
                                sub_col1, sub_col2, sub_col3, sub_col4 = st.columns([2, 2, 2, 3])
                                with sub_col1:
                                    st.write(f"**#{idx+1} {d['class'].title()}**")
                                with sub_col2:
                                    st.write(f"Category: `{d['category']}`")
                                with sub_col3:
                                    st.progress(d['confidence'])
                                with sub_col4:
                                    st.write(f"{d['confidence']*100:.1f}% confidence")

                    if show_raw_coords:
                        with st.expander("🛠️ Raw Bounding Box JSON"):
                            st.json(detections)

    # -------------------------------------------------------------------------
    # TAB 2: Batch Image Sorter
    # -------------------------------------------------------------------------
    with tab_batch:
        st.markdown("### 📦 Batch Waste Sorting")
        st.write("Upload a collection of photos (e.g. from a waste conveyor belt, sorting facility, or clean-up audit) to classify them in parallel.")

        batch_files = st.file_uploader(
            "Upload multiple waste images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if batch_files:
            if st.button(f"🚀 Classify All {len(batch_files)} Images", type="primary"):
                results_data = []
                category_counts = {"Recyclable": 0, "Biodegradable": 0, "Hazardous": 0, "General": 0, "Undetected": 0}
                
                prog_bar = st.progress(0)
                status_text = st.empty()

                processed_images = []

                for idx, file in enumerate(batch_files):
                    status_text.text(f"Processing image {idx+1}/{len(batch_files)}: {file.name}...")
                    try:
                        img = Image.open(file).convert("RGB")
                        if model is not None:
                            res = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                            dets = parse_detections(res.boxes, conf_threshold=conf_threshold)
                        else:
                            dets = []

                        if dets:
                            top = dets[0]
                            category_counts[top['category']] = category_counts.get(top['category'], 0) + 1
                            results_data.append({
                                "Filename": file.name,
                                "Detected Class": top['class'].title(),
                                "Category": top['category'],
                                "Confidence": f"{top['confidence']*100:.1f}%",
                                "Target Bin": DISPOSAL_INFO.get(top['class'], {}).get('bin', 'General')
                            })
                            annot = draw_styled_detections(img, dets, show_labels=True, show_conf=True)
                            processed_images.append((file.name, annot, f"{top['class'].title()} ({top['confidence']*100:.1f}%)"))
                        else:
                            category_counts["Undetected"] += 1
                            results_data.append({
                                "Filename": file.name,
                                "Detected Class": "Uncertain",
                                "Category": "Low Confidence",
                                "Confidence": "N/A",
                                "Target Bin": "Requires Manual Inspection"
                            })
                            processed_images.append((file.name, img, "Uncertain"))

                    except Exception as err:
                        results_data.append({
                            "Filename": file.name,
                            "Detected Class": "Error",
                            "Category": "Error",
                            "Confidence": str(err),
                            "Target Bin": "N/A"
                        })

                    prog_bar.progress((idx + 1) / len(batch_files))

                status_text.success(f"✅ Successfully processed {len(batch_files)} images!")

                # Summary Statistics
                st.markdown("#### 📊 Batch Sorting Summary")
                b_c1, b_c2, b_c3, b_c4, b_c5 = st.columns(5)
                b_c1.metric("Total Images", len(batch_files))
                b_c2.metric("♻️ Recyclable", category_counts.get("Recyclable", 0))
                b_c3.metric("🍏 Biodegradable", category_counts.get("Biodegradable", 0))
                b_c4.metric("☣️ Hazardous", category_counts.get("Hazardous", 0))
                b_c5.metric("🗑️ General / Uncertain", category_counts.get("General", 0) + category_counts.get("Undetected", 0))

                # Batch Table
                st.dataframe(results_data, use_container_width=True)

                # Batch Gallery Preview
                st.markdown("#### 🖼️ Visual Results Gallery")
                gallery_cols = st.columns(4)
                for g_idx, (fname, g_img, g_label) in enumerate(processed_images):
                    with gallery_cols[g_idx % 4]:
                        st.image(g_img, caption=f"{fname}\n{g_label}", use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Waste Recycling Directory & Guide
    # -------------------------------------------------------------------------
    with tab_directory:
        st.markdown("### 📚 Comprehensive Waste & Recycling Encyclopedia")
        st.write("Browse the 11 classes recognized by the AI model along with bin guidelines and contamination rules.")

        cat_filter = st.selectbox(
            "Filter by category:",
            ["All Categories", "Recyclable", "Biodegradable", "Hazardous", "General"]
        )

        grid_cols = st.columns(3)
        col_i = 0

        for c_name in CLASS_NAMES:
            c_cat = CATEGORY_MAP.get(c_name, 'General')
            if cat_filter != "All Categories" and c_cat != cat_filter:
                continue

            info = DISPOSAL_INFO.get(c_name, {})
            pill_class = f"pill-{c_cat.lower()}"

            with grid_cols[col_i % 3]:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.4rem; font-weight: 700;">{info.get('icon', '📦')} {c_name.title()}</span>
                        <span class="{pill_class}">{c_cat}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #475569; margin-bottom: 0.8rem;">
                        <b>Target Bin:</b> {info.get('bin', 'General Bin')}
                    </div>
                    <div style="font-size: 0.82rem; color: #334155; line-height: 1.4;">
                        <b>Disposal Tip:</b> {info.get('instructions', ['Clean and sort appropriately.'])[0]}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.8rem; border-top: 1px solid #f1f5f9; padding-top: 0.5rem;">
                        🌱 {info.get('impact', 'Promotes circular economy.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_i += 1

    # -------------------------------------------------------------------------
    # TAB 4: Model Architecture & Calibration
    # -------------------------------------------------------------------------
    with tab_diagnostics:
        st.markdown("### 📊 Architecture & Calibration")
        st.write("Examine the underlying YOLO weights, class distribution, and API integration details.")

        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            st.markdown("#### 🧠 Model Parameters")
            st.markdown(f"""
            - **Framework:** Ultralytics YOLOv8 / YOLOv11
            - **Weights File:** `{Path(model_info).name if model_info else 'best.pt'}`
            - **Active Confidence Threshold:** `{conf_threshold}`
            - **Active IOU Threshold:** `{iou_threshold}`
            - **Number of Detectable Classes:** `{len(CLASS_NAMES)}`
            - **Input Resolution:** `640 x 640 RGB`
            """)

            st.markdown("#### 📡 FastAPI Backend Integration")
            st.code("""
# FastAPI Endpoint Reference (main.py)
POST /predict       -> Single image detection
POST /predict/batch -> Multi-file batch inference
GET  /classes       -> List all 11 waste classes
GET  /health        -> Check model operational state
            """, language="python")

        with diag_c2:
            st.markdown("#### 🏷️ Class to Category Mapping")
            mapping_data = [
                {"Class ID": i, "Class Name": name, "Category": CATEGORY_MAP.get(name, 'General'), "Target Bin": DISPOSAL_INFO.get(name, {}).get('bin', 'General')}
                for i, name in enumerate(CLASS_NAMES)
            ]
            st.dataframe(mapping_data, use_container_width=True)


if __name__ == "__main__":
    main()
