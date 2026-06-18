import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import os
import shutil
import csv
import io
import logging

import models
from database import engine, get_db
from ml_pipeline import process_video_real
from config import CORS_ORIGINS

models.Base.metadata.create_all(bind=engine)

logger = logging.getLogger("violationvision")

app = FastAPI(title="ViolationVision MVP API")

# Configure CORS — origins come from config (env-var backed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Ensure upload dir exists
os.makedirs("uploads", exist_ok=True)

app.mount("/api/images", StaticFiles(directory="uploads"), name="images")

# Keep track of background processing tasks so they don't get garbage-collected
_background_tasks: set[asyncio.Task] = set()

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"

    # Chunked file copy to avoid loading entire video into RAM
    with open(file_location, "wb") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # Start background processing — store task reference to prevent GC
    task = asyncio.create_task(process_video_real(file_location, manager))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"info": f"file '{file.filename}' saved and processing started."}

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/violations")
def get_violations(db: Session = Depends(get_db)):
    violations = db.query(models.ViolationRecord).order_by(
        models.ViolationRecord.timestamp.desc()
    ).all()
    return violations

@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """Analytics — violation counts by type AND by severity."""
    violations = db.query(models.ViolationRecord).all()

    total = len(violations)
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}
    avg_confidence = 0.0

    for v in violations:
        by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        severity_key = v.severity or "MINOR"
        by_severity[severity_key] = by_severity.get(severity_key, 0) + 1
        avg_confidence += v.confidence if v.confidence else 0

    if total > 0:
        avg_confidence = round(avg_confidence / total, 3)

    return {
        "total_violations": total,
        "by_type": by_type,
        "by_severity": by_severity,
        "average_confidence": avg_confidence,
    }

@app.get("/api/violations/export")
def export_violations_csv(db: Session = Depends(get_db)):
    """Export all violation records as a downloadable CSV file."""
    violations = db.query(models.ViolationRecord).order_by(
        models.ViolationRecord.timestamp.desc()
    ).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Violation Type", "Severity", "Rider Count",
                      "Plate Number", "Confidence", "Image URL"])

    for v in violations:
        writer.writerow([v.id, v.timestamp, v.violation_type, v.severity,
                          v.rider_count, v.plate_number, v.confidence, v.image_url])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations_report.csv"}
    )
