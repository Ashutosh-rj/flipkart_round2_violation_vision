import asyncio
import cv2
import os
import time
import traceback
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
try:
    from torchvision.models import mobilenet_v3_small
except ImportError:
    pass
from preprocessing import preprocess_frame
import models
from database import SessionLocal
from config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    COCO_PERSON_CLASS,
    COCO_MOTORCYCLE_CLASS,
    COCO_CAR_CLASS,
    COCO_BUS_CLASS,
    COCO_TRUCK_CLASS,
    TRIPLE_RIDING_THRESHOLD,
    SPATIAL_OVERLAP_MARGIN_PX,
    SPATIAL_VERTICAL_TOLERANCE_FRACTION,
    VIOLATION_COOLDOWN_SECONDS,
    PARKING_TIME_THRESHOLD,
    HELMET_CONFIDENCE_THRESHOLD,
    SEVERITY_CRITICAL_RIDER_COUNT,
    SEVERITY_MAJOR_CONFIDENCE,
    TARGET_PROCESSING_FPS,
    TRAFFIC_LIGHT_STATE,
    TRAFFIC_FLOW_DIRECTION,
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

HELMET_MODEL_PATH = "helmet_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
helmet_classifier = None

if os.path.exists(HELMET_MODEL_PATH):
    try:
        helmet_classifier = mobilenet_v3_small(num_classes=2)
        helmet_classifier.load_state_dict(torch.load(HELMET_MODEL_PATH, map_location=device))
        helmet_classifier.to(device)
        helmet_classifier.eval()
        print(f"[Pipeline] Loaded PyTorch helmet classifier on {device}")
    except Exception as e:
        print(f"[Pipeline] Failed to load helmet classifier: {e}")

helmet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_severity(violation_type: str, rider_count: int, confidence: float) -> str:
    if violation_type == "Triple Riding":
        if rider_count >= SEVERITY_CRITICAL_RIDER_COUNT:
            return "CRITICAL"
        elif confidence >= SEVERITY_MAJOR_CONFIDENCE:
            return "MAJOR"
        else:
            return "MINOR"
    elif violation_type == "Helmet Non-compliance":
        return "MAJOR"
    elif violation_type == "Illegal Parking":
        return "MINOR"
    elif violation_type == "Wrong-side Driving":
        return "CRITICAL"
    elif violation_type == "Red-light Violation":
        return "CRITICAL"
    elif violation_type == "Stop-line Violation":
        return "MAJOR"
    elif violation_type == "Seatbelt Non-compliance":
        return "MINOR"
    return "MINOR"


def compute_composite_confidence(moto_conf: float, person_confs: list[float]) -> float:
    if not person_confs:
        return round(moto_conf, 3)
    avg_person = sum(person_confs) / len(person_confs)
    composite = 0.4 * moto_conf + 0.6 * avg_person
    return round(composite, 3)


def apply_nms(boxes, iou_threshold=0.8):
    filtered = []
    for item in sorted(boxes, key=lambda x: x[1], reverse=True):
        p_box = item[0]
        is_duplicate = False
        px1, py1, px2, py2 = p_box
        p_area = (px2 - px1) * (py2 - py1)
        
        for f_item in filtered:
            f_box = f_item[0]
            fx1, fy1, fx2, fy2 = f_box
            f_area = (fx2 - fx1) * (fy2 - fy1)
            inter_area = max(0, min(px2, fx2) - max(px1, fx1)) * max(0, min(py2, fy2) - max(py1, fy1))
            union_area = p_area + f_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > iou_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            filtered.append(item)
            
    return filtered


def is_rider_on_motorcycle(person_box, moto_box) -> bool:
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = moto_box

    inter_x1 = max(px1, mx1)
    inter_y1 = max(py1, my1)
    inter_x2 = min(px2, mx2)
    inter_y2 = min(py2, my2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    person_area = max(1, (px2 - px1) * (py2 - py1))
    moto_area = max(1, (mx2 - mx1) * (my2 - my1))
    
    # If the person's bounding box intersects with the motorcycle's bounding box
    # by at least 10% of the person's size OR 10% of the motorcycle's size, it's a rider.
    # This is extremely robust for CCTV top-down angles where geometric constraints fail.
    overlap_ratio_person = inter_area / person_area
    overlap_ratio_moto = inter_area / moto_area
    
    return overlap_ratio_person >= 0.10 or overlap_ratio_moto >= 0.10


def check_helmet(frame, person_box) -> bool:
    """
    Mock heuristic for helmet detection. 
    In production, use a dedicated helmet YOLO model.
    Here we crop the top 20% of the person (head region) and use an OpenCV heuristic 
    (or randomly flag based on confidence for MVP demonstration).
    We will use a simple heuristic: if the head region is too bright/skin-colored, no helmet.
    """
    px1, py1, px2, py2 = map(int, person_box)
    
    # Tighter crop: top 25% vertically, middle 50% horizontally
    width = px2 - px1
    height = py2 - py1
    
    head_left = px1 + int(width * 0.25)
    head_right = px2 - int(width * 0.25)
    head_bottom = py1 + int(height * 0.25)
    
    head_crop = frame[py1:head_bottom, head_left:head_right]
    
    if head_crop.size == 0:
        return True # Assume safe if can't see

    if helmet_classifier is not None:
        try:
            img_tensor = helmet_transform(head_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = helmet_classifier(img_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.item() == 1 # 1 = Helmet, 0 = No Helmet
        except Exception:
            pass # fallback to heuristic

    # Convert to HSV
    hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
    
    # Adjusted skin color range
    lower_skin = np.array([0, 15, 50], dtype=np.uint8)
    upper_skin = np.array([25, 170, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Adjusted dark hair color range (low brightness)
    lower_hair = np.array([0, 0, 0], dtype=np.uint8)
    upper_hair = np.array([180, 255, 80], dtype=np.uint8)
    mask_hair = cv2.inRange(hsv, lower_hair, upper_hair)
    
    mask = cv2.bitwise_or(mask_skin, mask_hair)
    
    skin_hair_ratio = cv2.countNonZero(mask) / (head_crop.shape[0] * head_crop.shape[1] + 1)
    
    # Print for debugging
    print(f"[Heuristic] Skin/Hair ratio: {skin_hair_ratio:.3f}")
    
    # Lower threshold to 2% (0.02). If even a tiny bit of skin or hair is visible, assume NO HELMET.
    has_helmet = skin_hair_ratio < 0.02
    return has_helmet


def check_seatbelt(frame, car_box) -> bool:
    """
    Mock heuristic for seatbelt detection using edge detection (HoughLines).
    In production, this requires a specialized YOLO model trained on seatbelts.
    This crops the windshield area of a car/truck and looks for a diagonal line.
    """
    cx1, cy1, cx2, cy2 = map(int, car_box)
    
    # Windshield is typically in the top 20% to 45% of the vehicle bounding box
    h = cy2 - cy1
    ws_top = cy1 + int(h * 0.2)
    ws_bottom = cy1 + int(h * 0.45)
    
    ws_crop = frame[ws_top:ws_bottom, cx1:cx2]
    if ws_crop.size == 0:
        return True # Default safe
        
    gray = cv2.cvtColor(ws_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            # Seatbelts are typically diagonal, around 30 to 60 degrees from horizontal
            if 30 < angle < 60:
                return True # Found a diagonal line = seatbelt

    # No diagonal lines found -> assume no seatbelt
    return False


def check_traffic_light_state(frame) -> str:
    """
    Robust heuristic for traffic light state.
    Looks at the top 30% of the frame for dominant red or green blobs.
    """
    h, w = frame.shape[:2]
    top_crop = frame[0:int(h*0.3), 0:w]
    
    if top_crop.size == 0:
        return "GREEN" # Safe default
        
    hsv = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    
    if red_pixels > green_pixels and red_pixels > 50:
        return "RED"
    return "GREEN"


def run_inference(frame):
    return model.predict(
        frame,
        classes=[COCO_PERSON_CLASS, COCO_MOTORCYCLE_CLASS, COCO_CAR_CLASS, COCO_BUS_CLASS, COCO_TRUCK_CLASS],
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
    )


def extract_plate(frame, moto_box):
    if not ocr:
        return "UNREADABLE"
    
    x1, y1, x2, y2 = map(int, moto_box)
    moto_h = y2 - y1
    # Try bottom 50% of motorcycle box (where plate usually is)
    crop_y1 = y1 + int(moto_h * 0.5)
    plate_crop = frame[crop_y1:y2, x1:x2]
    
    if plate_crop.size > 0:
        try:
            ocr_res = ocr.ocr(plate_crop, cls=True)
            if ocr_res and ocr_res[0]:
                candidate = ocr_res[0][0][1][0]
                conf = ocr_res[0][0][1][1]
                if conf > 0.6:
                    return candidate
        except Exception:
            pass
    return "UNREADABLE"


async def process_image_real(image_path: str, stop_line_y: int | None = None, camera_id: str = "cam_01"):
    """
    Process a single image for violations (Hackathon requirement)
    """
    db = SessionLocal()
    violations_found = []
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return {"error": "Could not read image"}
            
        enhanced_frame, _ = preprocess_frame(frame)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, run_inference, enhanced_frame)
        
        persons = []
        motorcycles = []
        other_vehicles = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                if cls == COCO_MOTORCYCLE_CLASS:
                    motorcycles.append((xyxy, conf))
                elif cls == COCO_PERSON_CLASS:
                    persons.append((xyxy, conf))
                elif cls in [COCO_CAR_CLASS, COCO_BUS_CLASS, COCO_TRUCK_CLASS]:
                    other_vehicles.append((xyxy, conf))
                    
        persons = apply_nms(persons, iou_threshold=0.8)
        motorcycles = apply_nms(motorcycles, iou_threshold=0.4)
        
        annotated = enhanced_frame.copy()
        
        # Check Motorcycles
        for moto_box, moto_conf in motorcycles:
            riders = []
            for p_box, p_conf in persons:
                if is_rider_on_motorcycle(p_box, moto_box):
                    riders.append((p_box, p_conf))
                    
            riders.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
            riders = riders[:3]
            
            rider_count = len(riders)
            
            # Check Triple Riding
            if rider_count >= TRIPLE_RIDING_THRESHOLD:
                plate = extract_plate(frame, moto_box)
                conf = compute_composite_confidence(moto_conf, [r[1] for r in riders])
                severity = classify_severity("Triple Riding", rider_count, conf)
                
                x1, y1, x2, y2 = map(int, moto_box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                for r_box, _ in riders:
                    rx1, ry1, rx2, ry2 = map(int, r_box)
                    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                    
                timestamp_dt = datetime.utcnow()
                filename = f"ev_tr_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                filepath = os.path.join("uploads", filename)
                cv2.imwrite(filepath, annotated)
                
                v_record = models.ViolationRecord(camera_id=camera_id,
                    timestamp=timestamp_dt,
                    violation_type="Triple Riding",
                    severity=severity,
                    rider_count=rider_count,
                    plate_number=plate,
                    confidence=conf,
                    image_url=f"/api/images/{filename}"
                )
                db.add(v_record)
                violations_found.append(v_record)
                
            # Check Helmet Non-compliance
            no_helmet_riders = []
            for r_box, r_conf in riders:
                if not check_helmet(frame, r_box):
                    no_helmet_riders.append((r_box, r_conf))
            
            if no_helmet_riders:
                plate = extract_plate(frame, moto_box)
                comp_conf = max(r[1] for r in no_helmet_riders)
                severity = classify_severity("Helmet Non-compliance", len(no_helmet_riders), comp_conf)
                
                annotated_helm = enhanced_frame.copy()
                x1, y1, x2, y2 = map(int, moto_box)
                cv2.rectangle(annotated_helm, (x1, y1), (x2, y2), (255, 165, 0), 3) # Orange moto
                
                for r_box, _ in no_helmet_riders:
                    rx1, ry1, rx2, ry2 = map(int, r_box)
                    cv2.rectangle(annotated_helm, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3) # Red rider
                    cv2.putText(annotated_helm, "No Helmet", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                timestamp_dt = datetime.utcnow()
                filename = f"ev_hl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                filepath = os.path.join("uploads", filename)
                cv2.imwrite(filepath, annotated_helm)
                
                v_record = models.ViolationRecord(camera_id=camera_id,
                    timestamp=timestamp_dt,
                    violation_type="Helmet Non-compliance",
                    severity=severity,
                    rider_count=len(no_helmet_riders),
                    plate_number=plate,
                    confidence=comp_conf,
                    image_url=f"/api/images/{filename}"
                )
                setattr(v_record, "detection_method", "heuristic")
                db.add(v_record)
                violations_found.append(v_record)
                    
        # Check Other Vehicles (Cars, Buses, Trucks)
        for v_box, v_conf in other_vehicles:
            vx1, vy1, vx2, vy2 = map(int, v_box)
            
            # Seatbelt check
            if not check_seatbelt(frame, v_box):
                severity = classify_severity("Seatbelt Non-compliance", 0, v_conf)
                annotated_sb = enhanced_frame.copy()
                cv2.rectangle(annotated_sb, (vx1, vy1), (vx2, vy2), (0, 165, 255), 3)
                cv2.putText(annotated_sb, "No Seatbelt", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                timestamp_dt = datetime.utcnow()
                filename = f"ev_sb_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                filepath = os.path.join("uploads", filename)
                cv2.imwrite(filepath, annotated_sb)
                
                v_record = models.ViolationRecord(camera_id=camera_id,
                    timestamp=timestamp_dt,
                    violation_type="Seatbelt Non-compliance",
                    severity=severity,
                    rider_count=0,
                    plate_number="UNREADABLE",
                    confidence=v_conf,
                    image_url=f"/api/images/{filename}"
                )
                setattr(v_record, "detection_method", "heuristic")
                db.add(v_record)
                violations_found.append(v_record)
            
            # Stop-line / Red-light check (Static image mock)
            # If the vehicle bottom crosses the stop_line_y and traffic is RED
            current_tl_state = check_traffic_light_state(frame)
            if stop_line_y is not None and vy2 > stop_line_y and current_tl_state == "RED":
                v_type = "Red-light Violation" if vy2 > stop_line_y + 100 else "Stop-line Violation"
                severity = classify_severity(v_type, 0, v_conf)
                annotated_sl = enhanced_frame.copy()
                
                # Draw virtual stop line
                cv2.line(annotated_sl, (0, stop_line_y), (annotated_sl.shape[1], stop_line_y), (0, 0, 255), 2)
                cv2.rectangle(annotated_sl, (vx1, vy1), (vx2, vy2), (255, 0, 0), 3)
                cv2.putText(annotated_sl, v_type, (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                timestamp_dt = datetime.utcnow()
                filename = f"ev_sl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                filepath = os.path.join("uploads", filename)
                cv2.imwrite(filepath, annotated_sl)
                
                v_record = models.ViolationRecord(camera_id=camera_id,
                    timestamp=timestamp_dt,
                    violation_type=v_type,
                    severity=severity,
                    rider_count=0,
                    plate_number="UNREADABLE",
                    confidence=v_conf,
                    image_url=f"/api/images/{filename}"
                )
                db.add(v_record)
                violations_found.append(v_record)

        # Skip illegal parking and wrong-side driving for single image processing 
        # as they require temporal data (tracking over time).
        
        db.commit()
        return [v.__dict__ for v in violations_found]
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        db.close()


async def process_video_real(video_path: str, websocket_manager, stop_line_y: int | None = None, camera_id: str = "cam_01"):
    db = SessionLocal()
    frames_processed = 0
    total_violations = 0

    try:
        print(f"[Pipeline] Starting for: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Pipeline] ERROR: Cannot open video file {video_path}")
            await websocket_manager.broadcast({"type": "error", "message": f"Cannot open video file: {os.path.basename(video_path)}"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_every_n = max(1, int(fps / TARGET_PROCESSING_FPS))

        frame_count = 0
        loop = asyncio.get_event_loop()

        next_moto_id = 0
        active_motos = {}  
        recent_violations = [] 
        
        # Tracker for Illegal Parking
        next_veh_id = 0
        active_vehicles = {} # veh_id -> {cx, cy, first_seen, last_seen}
        
        # Auto-adaptive global traffic flow direction tracker
        global_traffic_y_movement = 0
        dynamic_flow_direction = TRAFFIC_FLOW_DIRECTION

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
            enhanced_frame, _ = preprocess_frame(frame)
            results = await loop.run_in_executor(None, run_inference, enhanced_frame)

            raw_motorcycles = []
            persons = []
            raw_vehicles = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    if cls == COCO_MOTORCYCLE_CLASS:
                        raw_motorcycles.append((xyxy, conf))
                    elif cls == COCO_PERSON_CLASS:
                        persons.append((xyxy, conf))
                    elif cls in [COCO_CAR_CLASS, COCO_BUS_CLASS, COCO_TRUCK_CLASS]:
                        raw_vehicles.append((xyxy, conf))

            persons = apply_nms(persons, iou_threshold=0.8)
            raw_motorcycles = apply_nms(raw_motorcycles, iou_threshold=0.4)
            raw_vehicles = apply_nms(raw_vehicles, iou_threshold=0.5)

            current_video_time = frame_count / fps

            # --- TRACK MOTORCYCLES ---
            tracked_motorcycles = []
            used_mids = set()
            for moto_box, moto_conf in raw_motorcycles:
                mx1, my1, mx2, my2 = moto_box
                moto_cx, moto_cy = (mx1 + mx2) / 2, (my1 + my2) / 2
                best_id, best_dist = -1, float('inf')
                base_max_dist = max(mx2 - mx1, my2 - my1) * 3.0  
                
                for mid, (fcx, fcy, ft) in active_motos.items():
                    if mid in used_mids: continue
                    time_diff = current_video_time - ft
                    if time_diff > 15.0: continue
                    # Substantially increase dynamic search radius for robust tracking
                    dynamic_max_dist = max(base_max_dist + (time_diff * 800.0), 800.0)
                    dist = ((moto_cx - fcx) ** 2 + (moto_cy - fcy) ** 2) ** 0.5
                    if dist < best_dist and dist < dynamic_max_dist:
                        best_dist = dist
                        best_id = mid

                if best_id == -1:
                    best_id = next_moto_id
                    next_moto_id += 1

                used_mids.add(best_id)
                active_motos[best_id] = (moto_cx, moto_cy, current_video_time)
                tracked_motorcycles.append((moto_box, moto_conf, best_id))

            active_motos = {mid: data for mid, data in active_motos.items() if current_video_time - data[2] <= 15.0}

            # --- TRACK OTHER VEHICLES (ILLEGAL PARKING) ---
            used_vids = set()
            for v_box, v_conf in raw_vehicles:
                vx1, vy1, vx2, vy2 = v_box
                vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
                best_id, best_dist = -1, float('inf')
                
                for vid, vdata in active_vehicles.items():
                    if vid in used_vids: continue
                    time_diff = current_video_time - vdata['last_seen']
                    if time_diff > 15.0: continue
                    dist = ((vcx - vdata['cx']) ** 2 + (vcy - vdata['cy']) ** 2) ** 0.5
                    # Larger search radius for cars
                    if dist < best_dist and dist < 1000:
                        best_dist = dist
                        best_id = vid

                if best_id == -1:
                    best_id = next_veh_id
                    next_veh_id += 1
                    active_vehicles[best_id] = {'cx': vcx, 'cy': vcy, 'first_seen': current_video_time, 'last_seen': current_video_time, 'box': v_box, 'conf': v_conf}
                else:
                    active_vehicles[best_id]['cx'] = vcx
                    active_vehicles[best_id]['cy'] = vcy
                    active_vehicles[best_id]['last_seen'] = current_video_time
                    active_vehicles[best_id]['box'] = v_box
                    active_vehicles[best_id]['conf'] = v_conf
                used_vids.add(best_id)
                
            active_vehicles = {vid: data for vid, data in active_vehicles.items() if current_video_time - data['last_seen'] <= 15.0}


            # --- CHECK VIOLATIONS ---
            # 1. Illegal Parking
            for vid, vdata in active_vehicles.items():
                if current_video_time - vdata['first_seen'] >= PARKING_TIME_THRESHOLD:
                    # Check if already alerted
                    is_duplicate = False
                    for (v_time, v_id_recorded, v_type) in recent_violations:
                        if v_id_recorded == vid and v_type == "Illegal Parking" and current_video_time - v_time <= VIOLATION_COOLDOWN_SECONDS:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        plate_number = extract_plate(frame, vdata['box'])
                        timestamp_dt = datetime.utcnow()
                        filename = f"ev_park_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                        filepath = os.path.join("uploads", filename)
                        
                        annotated = enhanced_frame.copy()
                        vx1, vy1, vx2, vy2 = map(int, vdata['box'])
                        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 255), 3) # Magenta for parking
                        cv2.putText(annotated, "Illegal Parking", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        cv2.imwrite(filepath, annotated)

                        new_record = models.ViolationRecord(
                            timestamp=timestamp_dt, camera_id=camera_id, violation_type="Illegal Parking",
                            severity="MINOR", rider_count=0, plate_number=plate_number,
                            confidence=vdata['conf'], image_url=f"/api/images/{filename}"
                        )
                        db.add(new_record)
                        total_violations += 1
                        recent_violations.append((current_video_time, vid, "Illegal Parking"))
                        
                        await websocket_manager.broadcast({
                            "camera_id": camera_id, "type": "violation", "violation_type": "Illegal Parking", "severity": "MINOR",
                            "plate_number": plate_number, "confidence": float(vdata['conf']), "rider_count": 0,
                            "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                        })
                        # Reset first_seen to prevent spamming while it's still parked
                        vdata['first_seen'] = current_video_time 


            # 2. Seatbelt, Wrong-side, Stop-line, Red-light
            for vid, vdata in active_vehicles.items():
                vx1, vy1, vx2, vy2 = map(int, vdata['box'])
                v_conf = vdata['conf']
                plate_number = extract_plate(frame, vdata['box'])
                
                # Check Seatbelt
                if not check_seatbelt(frame, vdata['box']):
                    is_dup = any((v_id_recorded == vid and v_type == "Seatbelt Non-compliance" and current_video_time - vt <= VIOLATION_COOLDOWN_SECONDS) for vt, v_id_recorded, v_type in recent_violations)
                    if not is_dup:
                        recent_violations.append((current_video_time, vid, "Seatbelt Non-compliance"))
                        severity = classify_severity("Seatbelt Non-compliance", 0, v_conf)
                        
                        timestamp_dt = datetime.utcnow()
                        filename = f"ev_sb_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                        filepath = os.path.join("uploads", filename)
                        
                        annotated = enhanced_frame.copy()
                        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 165, 255), 3)
                        cv2.putText(annotated, f"No Seatbelt | {severity}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.imwrite(filepath, annotated)

                        new_record = models.ViolationRecord(
                            timestamp=timestamp_dt, camera_id=camera_id, violation_type="Seatbelt Non-compliance", severity=severity,
                            rider_count=0, plate_number=plate_number, confidence=v_conf, image_url=f"/api/images/{filename}"
                        )
                        db.add(new_record)
                        total_violations += 1
                        
                        await websocket_manager.broadcast({
                            "camera_id": camera_id, "type": "violation", "violation_type": "Seatbelt Non-compliance", "severity": severity,
                            "plate_number": plate_number, "confidence": float(v_conf), "rider_count": 0,
                            "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat(),
                            "detection_method": "heuristic"
                        })

                # Check Wrong-side driving
                if 'first_cy' not in vdata:
                    vdata['first_cy'] = vdata['cy']
                
                movement_y = vdata['cy'] - vdata['first_cy']
                
                # Auto-adaptive flow update
                if 'last_cy' in vdata:
                    global_traffic_y_movement += (vdata['cy'] - vdata['last_cy'])
                vdata['last_cy'] = vdata['cy']
                
                # Dynamically update the assumed traffic flow direction
                if global_traffic_y_movement > 500:
                    dynamic_flow_direction = "down"
                elif global_traffic_y_movement < -500:
                    dynamic_flow_direction = "up"
                    
                is_wrong_side = False
                if dynamic_flow_direction == "down" and movement_y < -50:
                    is_wrong_side = True
                elif dynamic_flow_direction == "up" and movement_y > 50:
                    is_wrong_side = True
                if is_wrong_side:
                    is_dup = any((v_id_recorded == vid and v_type == "Wrong-side Driving" and current_video_time - vt <= VIOLATION_COOLDOWN_SECONDS) for vt, v_id_recorded, v_type in recent_violations)
                    if not is_dup:
                        recent_violations.append((current_video_time, vid, "Wrong-side Driving"))
                        severity = classify_severity("Wrong-side Driving", 0, v_conf)
                        
                        timestamp_dt = datetime.utcnow()
                        filename = f"ev_ws_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                        filepath = os.path.join("uploads", filename)
                        
                        annotated = enhanced_frame.copy()
                        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 0, 0), 3) # Black box
                        cv2.putText(annotated, f"Wrong Way! | {severity}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imwrite(filepath, annotated)

                        new_record = models.ViolationRecord(
                            timestamp=timestamp_dt, camera_id=camera_id, violation_type="Wrong-side Driving", severity=severity,
                            rider_count=0, plate_number=plate_number, confidence=v_conf, image_url=f"/api/images/{filename}"
                        )
                        db.add(new_record)
                        total_violations += 1
                        
                        await websocket_manager.broadcast({
                            "camera_id": camera_id, "type": "violation", "violation_type": "Wrong-side Driving", "severity": severity,
                            "plate_number": plate_number, "confidence": float(v_conf), "rider_count": 0,
                            "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                        })

                # Check Stop-line & Red-light
                # We assume the vehicle crosses the horizontal line from above (y increases).
                current_tl_state = check_traffic_light_state(frame)
                if stop_line_y is not None and current_tl_state == "RED":
                    if vy2 > stop_line_y:
                        v_type = "Red-light Violation" if vy2 > stop_line_y + 100 else "Stop-line Violation"
                        is_dup = any((v_id_recorded == vid and vt_type == v_type and current_video_time - vt <= VIOLATION_COOLDOWN_SECONDS) for vt, v_id_recorded, vt_type in recent_violations)
                        
                        if not is_dup:
                            recent_violations.append((current_video_time, vid, v_type))
                            severity = classify_severity(v_type, 0, v_conf)
                            
                            timestamp_dt = datetime.utcnow()
                            filename = f"ev_sl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            
                            annotated = enhanced_frame.copy()
                            cv2.line(annotated, (0, stop_line_y), (annotated.shape[1], stop_line_y), (0, 0, 255), 2)
                            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 3)
                            cv2.putText(annotated, f"{v_type} | {severity}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            cv2.imwrite(filepath, annotated)

                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type=v_type, severity=severity,
                                rider_count=0, plate_number=plate_number, confidence=v_conf, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": v_type, "severity": severity,
                                "plate_number": plate_number, "confidence": float(v_conf), "rider_count": 0,
                                "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })

            # 3. Triple Riding & Helmet Non-compliance
            for moto_box, moto_conf, moto_id in tracked_motorcycles:
                mx1, my1, mx2, my2 = moto_box
                riders_on_moto = []

                for p_box, p_conf in persons:
                    if is_rider_on_motorcycle(p_box, moto_box):
                        riders_on_moto.append((p_box, p_conf))
                
                riders_on_moto.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
                riders_on_moto = riders_on_moto[:3]
                rider_count = len(riders_on_moto)
                
                # Triple Riding
                if rider_count >= TRIPLE_RIDING_THRESHOLD:
                    plate_number = extract_plate(frame, moto_box)
                    is_duplicate = any((v_id_recorded == moto_id and v_type == "Triple Riding" and current_video_time - vt <= VIOLATION_COOLDOWN_SECONDS) for vt, v_id_recorded, v_type in recent_violations)
                    
                    if not is_duplicate:
                        recent_violations.append((current_video_time, moto_id, "Triple Riding"))
                        comp_conf = compute_composite_confidence(moto_conf, [r[1] for r in riders_on_moto])
                        severity = classify_severity("Triple Riding", rider_count, comp_conf)
                        
                        timestamp_dt = datetime.utcnow()
                        filename = f"ev_tr_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                        filepath = os.path.join("uploads", filename)

                        annotated = enhanced_frame.copy()
                        x1, y1, x2, y2 = map(int, moto_box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        for r_box, _ in riders_on_moto:
                            rx1, ry1, rx2, ry2 = map(int, r_box)
                            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                        cv2.putText(annotated, f"Triple Riding | {severity}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imwrite(filepath, annotated)

                        new_record = models.ViolationRecord(
                            timestamp=timestamp_dt, camera_id=camera_id, violation_type="Triple Riding", severity=severity,
                            rider_count=rider_count, plate_number=plate_number, confidence=comp_conf,
                            image_url=f"/api/images/{filename}"
                        )
                        db.add(new_record)
                        total_violations += 1
                        
                        await websocket_manager.broadcast({
                            "camera_id": camera_id, "type": "violation", "violation_type": "Triple Riding", "severity": severity,
                            "plate_number": plate_number, "confidence": float(comp_conf), "rider_count": rider_count,
                            "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                        })

                # Helmet Non-compliance
                no_helmet_riders = []
                for r_box, r_conf in riders_on_moto:
                    if not check_helmet(frame, r_box):
                        no_helmet_riders.append((r_box, r_conf))

                if no_helmet_riders:
                    plate_number = extract_plate(frame, moto_box)
                    # Avoid duplicate helmet alerts for the same motorcycle quickly
                    is_duplicate = any((v_id_recorded == moto_id and v_type == "Helmet Non-compliance" and current_video_time - vt <= VIOLATION_COOLDOWN_SECONDS) for vt, v_id_recorded, v_type in recent_violations)
                    
                    if not is_duplicate:
                        recent_violations.append((current_video_time, moto_id, "Helmet Non-compliance"))
                        
                        # Use max confidence of those without helmet
                        comp_conf = max(r[1] for r in no_helmet_riders)
                        severity = classify_severity("Helmet Non-compliance", len(no_helmet_riders), comp_conf)
                        
                        timestamp_dt = datetime.utcnow()
                        filename = f"ev_hl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                        filepath = os.path.join("uploads", filename)

                        annotated = enhanced_frame.copy()
                        x1, y1, x2, y2 = map(int, moto_box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 3)
                        
                        for r_box, _ in no_helmet_riders:
                            rx1, ry1, rx2, ry2 = map(int, r_box)
                            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
                            cv2.putText(annotated, f"No Helmet", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                        cv2.imwrite(filepath, annotated)

                        new_record = models.ViolationRecord(
                            timestamp=timestamp_dt, camera_id=camera_id, violation_type="Helmet Non-compliance", severity=severity,
                            rider_count=len(no_helmet_riders), plate_number=plate_number, confidence=comp_conf,
                            image_url=f"/api/images/{filename}"
                        )
                        db.add(new_record)
                        total_violations += 1
                        
                        await websocket_manager.broadcast({
                            "camera_id": camera_id, "type": "violation", "violation_type": "Helmet Non-compliance", "severity": severity,
                            "plate_number": plate_number, "confidence": float(comp_conf), "rider_count": len(no_helmet_riders),
                            "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat(),
                            "detection_method": "heuristic"
                        })

            db.commit()
            recent_violations = [v for v in recent_violations if current_video_time - v[0] <= VIOLATION_COOLDOWN_SECONDS]

            await asyncio.sleep(0.01)

        cap.release()
        await websocket_manager.broadcast({
            "type": "complete",
            "message": f"Processing complete: {frames_processed} frames analyzed, {total_violations} violations found",
            "frames_processed": frames_processed, "total_violations": total_violations
        })

    except Exception as e:
        print(f"[Pipeline] FATAL: {str(e)}")
        traceback.print_exc()
        try:
            await websocket_manager.broadcast({"type": "error", "message": f"Pipeline error: {str(e)}"})
        except: pass
    finally:
        db.close()
