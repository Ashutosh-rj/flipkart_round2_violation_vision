"""
Centralized configuration for ViolationVision MVP.
All tuneable parameters and environment-based settings live here.
"""
import os

# ── Database ──────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

# ── CORS ──────────────────────────────────────────────────────────────
# Comma-separated origins, e.g. "http://localhost:3000,https://myapp.com"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── YOLO Model ────────────────────────────────────────────────────────
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.4"))

# COCO class IDs used by the pipeline
COCO_PERSON_CLASS = 0
COCO_MOTORCYCLE_CLASS = 3

# ── Preprocessing ─────────────────────────────────────────────────────
YOLO_INPUT_SIZE = (640, 640)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

# ── Detection Logic ──────────────────────────────────────────────────
# Minimum riders on a motorcycle to trigger a violation
TRIPLE_RIDING_THRESHOLD = 3

# Horizontal margin (pixels) when checking person ↔ motorcycle overlap
SPATIAL_OVERLAP_MARGIN_PX = 20

# Vertical tolerance: person center must be within this fraction of
# motorcycle height ABOVE the motorcycle top to still count as a rider.
# 0.3 means up to 30 % of the moto height above its top edge.
SPATIAL_VERTICAL_TOLERANCE_FRACTION = 0.3

# Seconds to suppress duplicate alerts for the same vehicle.
# Must be long enough that a vehicle fully exits the camera's field of view.
VIOLATION_COOLDOWN_SECONDS = float(os.getenv("VIOLATION_COOLDOWN", "30.0"))

# ── Severity Classification ──────────────────────────────────────────
# rider_count >= this ⇒ CRITICAL
SEVERITY_CRITICAL_RIDER_COUNT = 4

# If rider_count < CRITICAL threshold but confidence >= this ⇒ MAJOR
SEVERITY_MAJOR_CONFIDENCE = 0.7

# ── Frame Sampling ────────────────────────────────────────────────────
# Target frames per second to process (independent of source FPS)
TARGET_PROCESSING_FPS = 3
