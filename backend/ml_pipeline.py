import asyncio
import cv2
import os
import time
import traceback
from datetime import datetime
from ultralytics import YOLO
from preprocessing import preprocess_frame
import models
from database import SessionLocal
from config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    COCO_PERSON_CLASS,
    COCO_MOTORCYCLE_CLASS,
    TRIPLE_RIDING_THRESHOLD,
    SPATIAL_OVERLAP_MARGIN_PX,
    SPATIAL_VERTICAL_TOLERANCE_FRACTION,
    VIOLATION_COOLDOWN_SECONDS,
    SEVERITY_CRITICAL_RIDER_COUNT,
    SEVERITY_MAJOR_CONFIDENCE,
    TARGET_PROCESSING_FPS,
)

# Lazy-load PaddleOCR
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
except ImportError:
    ocr = None
    print("Warning: PaddleOCR not available. Plates will be marked as UNREADABLE.")

# Load YOLO model once at module level
model = YOLO(YOLO_MODEL_PATH)


def classify_severity(violation_type: str, rider_count: int, confidence: float) -> str:
    """
    Classify violation severity based on rider count AND confidence.

    - 4+ riders                          → CRITICAL
    - 3 riders with high confidence (≥ threshold) → MAJOR
    - 3 riders with low confidence        → MINOR  (was previously dead-branch bug)
    """
    if violation_type == "Triple Riding":
        if rider_count >= SEVERITY_CRITICAL_RIDER_COUNT:
            return "CRITICAL"
        elif confidence >= SEVERITY_MAJOR_CONFIDENCE:
            return "MAJOR"
        else:
            return "MINOR"
    return "MINOR"


def compute_composite_confidence(moto_conf: float, person_confs: list[float]) -> float:
    """
    Compute a composite confidence score from motorcycle + person detections.
    Weighted average: motorcycle contributes 40%, average person confidence 60%.
    """
    if not person_confs:
        return round(moto_conf, 3)
    avg_person = sum(person_confs) / len(person_confs)
    composite = 0.4 * moto_conf + 0.6 * avg_person
    return round(composite, 3)


def apply_nms(boxes, iou_threshold=0.8):
    """
    Remove redundant overlapping bounding detections (e.g. same object detected twice).
    Uses standard IoU to suppress duplicate boxes.
    """
    filtered = []
    for p_box, p_conf in sorted(boxes, key=lambda x: x[1], reverse=True):
        is_duplicate = False
        px1, py1, px2, py2 = p_box
        p_area = (px2 - px1) * (py2 - py1)
        
        for f_box, _ in filtered:
            fx1, fy1, fx2, fy2 = f_box
            f_area = (fx2 - fx1) * (fy2 - fy1)
            inter_area = max(0, min(px2, fx2) - max(px1, fx1)) * max(0, min(py2, fy2) - max(py1, fy1))
            union_area = p_area + f_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > iou_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            filtered.append((p_box, p_conf))
            
    return filtered


def is_rider_on_motorcycle(person_box, moto_box) -> bool:
    """
    Determine whether a detected person is spatially riding a motorcycle.
    Uses strict geometric checks to avoid counting background pedestrians.
    """
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = moto_box

    px_center = (px1 + px2) / 2
    p_bottom = py2

    moto_width = mx2 - mx1
    moto_height = my2 - my1

    # 1. Horizontal Check: Person's center must be within the motorcycle's width (plus a tiny 10% margin)
    margin_x = moto_width * 0.1
    horizontal_ok = (mx1 - margin_x) <= px_center <= (mx2 + margin_x)

    # 2. Vertical Bottom Check: Rider's feet/hips (bottom of bounding box) must be near the motorcycle body.
    # It cannot be above the motorcycle (flying) or way below the motorcycle (standing in front).
    margin_y_bottom = moto_height * 0.2
    vertical_ok = my1 <= p_bottom <= (my2 + margin_y_bottom)
    
    # 3. Size Check: Person cannot be a tiny speck in the background.
    person_height = py2 - py1
    size_ok = person_height >= (moto_height * 0.3)

    # 4. Overlap Check: A significant portion of the person MUST be inside the motorcycle.
    # This completely eliminates pedestrians in the background who just happen to align on the X-axis.
    inter_x1 = max(px1, mx1)
    inter_y1 = max(py1, my1)
    inter_x2 = min(px2, mx2)
    inter_y2 = min(py2, my2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    person_area = max(1, (px2 - px1) * (py2 - py1))
    
    # At least 15% of the person's bounding box must be inside the motorcycle's bounding box
    overlap_ratio = inter_area / person_area
    overlap_ok = overlap_ratio >= 0.15

    return horizontal_ok and vertical_ok and size_ok and overlap_ok


def run_inference(frame):
    """
    Synchronous YOLO inference — called via run_in_executor to avoid
    blocking the async event loop.
    """
    return model.predict(
        frame,
        classes=[COCO_PERSON_CLASS, COCO_MOTORCYCLE_CLASS],
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
    )


async def process_video_real(video_path: str, websocket_manager):
    """
    Real ML pipeline: opens video, runs YOLO, checks spatial overlap
    for Triple Riding, runs OCR on plates, saves annotated evidence,
    persists to database, and broadcasts via WebSocket.
    """
    # Create our OWN database session (not the request's session, which closes)
    db = SessionLocal()

    frames_processed = 0
    total_violations = 0

    try:
        print(f"[Pipeline] Starting for: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Pipeline] ERROR: Cannot open video file {video_path}")
            await websocket_manager.broadcast({
                "type": "error",
                "message": f"Cannot open video file: {os.path.basename(video_path)}"
            })
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_every_n = max(1, int(fps / TARGET_PROCESSING_FPS))

        frame_count = 0
        loop = asyncio.get_event_loop()

        # Lightweight object tracker: track ALL motorcycles across frames
        next_moto_id = 0
        active_motos = {}  # moto_id -> (cx, cy, last_seen_time)
        flagged_motos = {} # moto_id -> last_violation_time
        recent_violations = [] # List of tuples: (time, plate_number)
        POSITION_MATCH_FRACTION = 0.25  # Max movement allowed between frames
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

        # Notify frontend that processing has started
        await websocket_manager.broadcast({
            "type": "status",
            "message": f"Processing started: {total_frames} frames @ {fps:.0f} FPS",
            "total_frames": total_frames
        })

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % process_every_n != 0:
                continue

            frames_processed += 1

            # 1. Preprocess
            enhanced_frame, raw_resized = preprocess_frame(frame)

            # 2. Run YOLO in a thread pool so we don't block the event loop
            results = await loop.run_in_executor(None, run_inference, enhanced_frame)

            # 3. Parse detections
            raw_motorcycles = []
            persons = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    if cls == COCO_MOTORCYCLE_CLASS:
                        raw_motorcycles.append((xyxy, conf))
                    elif cls == COCO_PERSON_CLASS:
                        persons.append((xyxy, conf))

            # Deduplicate bounding boxes
            persons = apply_nms(persons, iou_threshold=0.8)
            raw_motorcycles = apply_nms(raw_motorcycles, iou_threshold=0.4)

            current_video_time = frame_count / fps

            # 4. Update Tracker
            tracked_motorcycles = []
            used_mids = set()
            for moto_box, moto_conf in raw_motorcycles:
                mx1, my1, mx2, my2 = moto_box
                moto_cx = (mx1 + mx2) / 2
                moto_cy = (my1 + my2) / 2
                moto_width = mx2 - mx1
                moto_height = my2 - my1

                best_id = -1
                best_dist = float('inf')

                # Allow moving up to 3x its size, plus extra distance based on how long it was lost
                base_max_dist = max(moto_width, moto_height) * 3.0  
                for mid, (fcx, fcy, ft) in active_motos.items():
                    if mid in used_mids:
                        continue
                    
                    time_diff = current_video_time - ft
                    # Keep tracks alive for up to 15 seconds to survive long occlusions
                    if time_diff > 15.0:
                        continue
                    
                    # Expand search radius by 200 pixels per second of occlusion
                    dynamic_max_dist = base_max_dist + (time_diff * 200.0)
                    
                    dist = ((moto_cx - fcx) ** 2 + (moto_cy - fcy) ** 2) ** 0.5
                    if dist < best_dist and dist < dynamic_max_dist:
                        best_dist = dist
                        best_id = mid

                # If there are no active tracks nearby, assume it's a new vehicle.
                if best_id == -1:
                    best_id = next_moto_id
                    next_moto_id += 1

                used_mids.add(best_id)
                active_motos[best_id] = (moto_cx, moto_cy, current_video_time)
                tracked_motorcycles.append((moto_box, moto_conf, best_id))

            # Cleanup old tracks
            active_motos = {mid: data for mid, data in active_motos.items() if current_video_time - data[2] <= 15.0}

            # 5. Check Violations
            for moto_box, moto_conf, moto_id in tracked_motorcycles:
                mx1, my1, mx2, my2 = moto_box
                moto_height = my2 - my1
                moto_width = mx2 - mx1

                rider_count = 0
                rider_confs = []
                riders_on_moto = []

                for p_box, p_conf in persons:
                    if is_rider_on_motorcycle(p_box, moto_box):
                        riders_on_moto.append(p_box)
                        rider_confs.append(p_conf)
                
                # Keep only the 3 largest bounding boxes!
                # This explicitly ignores smaller false-positives like backpacks
                # and fixes the "4 riders" problem robustly without risking merging actual riders.
                def get_area(box):
                    return (box[2] - box[0]) * (box[3] - box[1])
                
                # Zip boxes and confs so we can sort them together
                paired = list(zip(riders_on_moto, rider_confs))
                paired.sort(key=lambda x: get_area(x[0]), reverse=True)
                paired = paired[:3]  # keep top 3
                
                riders_on_moto = [p[0] for p in paired]
                rider_confs = [p[1] for p in paired]
                
                rider_count = len(riders_on_moto)
                
                if rider_count >= TRIPLE_RIDING_THRESHOLD:
                    x1, y1, x2, y2 = map(int, moto_box)
                    moto_h = y2 - y1
                    moto_w = x2 - x1
                    
                    # Perform OCR FIRST so we can check if plate became readable
                    plate_number = "UNREADABLE"
                    if ocr:
                        try:
                            # Try bottom 50% of motorcycle box (where plate usually is)
                            crop_y1 = y1 + int(moto_h * 0.5)
                            plate_crop = frame[crop_y1:y2, x1:x2]
                            
                            if plate_crop.size > 0:
                                ocr_result = ocr.ocr(plate_crop, cls=False)
                                if ocr_result and ocr_result[0]:
                                    texts = [line[1][0] for line in ocr_result[0] if line[1][1] > 0.4]
                                    if texts:
                                        raw_text = "".join(texts).replace(" ", "").upper()
                                        raw_text = ''.join(c for c in raw_text if c.isalnum())
                                        if len(raw_text) >= 4:
                                            plate_number = raw_text
                        except Exception as e:
                            print(f"[Pipeline] OCR Error: {e}")
                            
                    # Check if THIS specific motorcycle was already flagged recently
                    last_violation_data = flagged_motos.get(moto_id, (-999, "UNREADABLE"))
                    last_time = last_violation_data[0]
                    last_plate = last_violation_data[1]
                    
                    # -----------------------------------------------------
                    # VIOLATION MEMORY: Suppress Duplicate Alerts
                    # -----------------------------------------------------
                    is_duplicate_violation = False
                    for (v_time, v_plate) in recent_violations:
                        if current_video_time - v_time <= 60.0:
                            # It's the same triple riding vehicle
                            if plate_number != "UNREADABLE" and v_plate == "UNREADABLE":
                                # Plate became readable! Allow this update to go through
                                # (We will update the list below)
                                pass
                            else:
                                # Either still unreadable, or already read. Suppress!
                                is_duplicate_violation = True
                                break
                    
                    if is_duplicate_violation:
                        print(f"[Pipeline] Frame {frame_count}: SUPPRESSED duplicate Triple Riding alert.")
                        continue
                        
                    # Update our recent violations memory
                    # (Keep the list small by only storing recent ones)
                    recent_violations = [(vt, vp) for (vt, vp) in recent_violations if current_video_time - vt <= 60.0]
                    recent_violations.append((current_video_time, plate_number))

                    # -----------------------------------------------------
                    # End Violation Memory
                    # -----------------------------------------------------

                    # Composite confidence from motorcycle + person scores
                    composite_conf = compute_composite_confidence(moto_conf, rider_confs)

                    # Severity classification
                    severity = classify_severity("Triple Riding", rider_count, composite_conf)

                    timestamp_dt = datetime.utcnow()
                    timestamp_str = timestamp_dt.isoformat()
                    filename = f"evidence_{timestamp_str.replace(':', '-')}.jpg"
                    filepath = os.path.join("uploads", filename)

                    # Draw bounding boxes on a copy so multiple violations
                    # don't overwrite each other's annotations
                    annotated = enhanced_frame.copy()
                    x1, y1, x2, y2 = map(int, moto_box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Draw green boxes around all detected riders to provide visual proof
                    for r_box in riders_on_moto:
                        rx1, ry1, rx2, ry2 = map(int, r_box)
                        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                        
                    label = f"Triple Riding | {severity} | {composite_conf:.0%}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


                    # Save annotated evidence image
                    cv2.imwrite(filepath, annotated)

                    # Persist to database
                    new_record = models.ViolationRecord(
                        timestamp=timestamp_dt,
                        violation_type="Triple Riding",
                        severity=severity,
                        rider_count=rider_count,
                        plate_number=plate_number,
                        confidence=composite_conf,
                        image_url=f"/api/images/{filename}"
                    )
                    db.add(new_record)
                    db.commit()

                    total_violations += 1

                    violation_payload = {
                        "type": "violation",
                        "violation_type": "Triple Riding",
                        "severity": severity,
                        "plate_number": plate_number,
                        "confidence": float(composite_conf),
                        "rider_count": rider_count,
                        "image_url": f"/api/images/{filename}",
                        "timestamp": timestamp_str
                    }

                    print(f"[Pipeline] Violation #{total_violations}: Triple Riding | {severity} | conf={composite_conf} | plate={plate_number} | pos=({moto_cx:.0f},{moto_cy:.0f})")
                    await websocket_manager.broadcast(violation_payload)

            # Yield to event loop
            await asyncio.sleep(0.01)

        cap.release()

        # Send completion message
        await websocket_manager.broadcast({
            "type": "complete",
            "message": f"Processing complete: {frames_processed} frames analyzed, {total_violations} violations found",
            "frames_processed": frames_processed,
            "total_violations": total_violations
        })
        print(f"[Pipeline] Done: {frames_processed} frames, {total_violations} violations")

    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        print(f"[Pipeline] FATAL: {error_msg}")
        traceback.print_exc()
        try:
            await websocket_manager.broadcast({
                "type": "error",
                "message": error_msg
            })
        except Exception:
            pass
    finally:
        db.close()
