import os
import cv2
import uuid
import math
import asyncio
import traceback
import re
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
try:
    from torchvision.models import mobilenet_v3_small
except ImportError:
    pass
from paddleocr import PaddleOCR
from deep_sort_realtime.deepsort_tracker import DeepSort

from preprocessing import preprocess_frame
import models
from database import SessionLocal
from config import YOLO_CONFIDENCE_THRESHOLD, YOLO_MODEL_PATH, TRIPLE_RIDING_THRESHOLD, VIOLATION_COOLDOWN_SECONDS

class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
    def extract_license_plate(self, frame, vehicle_box) -> str:
        px1, py1, px2, py2 = map(int, vehicle_box)
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return "UNREADABLE"
        
        try:
            results = self.ocr.ocr(crop, cls=True)
            if not results or not results[0]:
                return "UNREADABLE"
            
            texts = [line[1][0] for line in results[0]]
            plate_text = "".join(texts).replace(" ", "").upper()
            plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)
            
            if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$', plate_text):
                return plate_text
            elif len(plate_text) >= 4 and re.search(r'[A-Z]{2}.*[0-9]{4}', plate_text):
                return plate_text
            elif len(plate_text) >= 4:
                return f"{plate_text}"
        except Exception:
            pass
            
        return "UNREADABLE"

class ViolationDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.helmet_classifier = None
        self._load_helmet_model()
        self.helmet_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_helmet_model(self):
        HELMET_MODEL_PATH = "helmet_model.pt"
        if os.path.exists(HELMET_MODEL_PATH):
            try:
                self.helmet_classifier = mobilenet_v3_small(num_classes=2)
                self.helmet_classifier.load_state_dict(torch.load(HELMET_MODEL_PATH, map_location=self.device))
                self.helmet_classifier.to(self.device)
                self.helmet_classifier.eval()
                print(f"[ViolationDetector] Loaded PyTorch helmet classifier on {self.device}")
            except Exception as e:
                print(f"[ViolationDetector] Failed to load helmet classifier: {e}")

    def classify_severity(self, violation_type: str, rider_count: int, confidence: float) -> str:
        if violation_type == "Triple Riding":
            return "CRITICAL" if rider_count > 3 else "MAJOR"
        if violation_type == "Helmet Non-compliance":
            return "MAJOR" if rider_count > 1 else "MINOR"
        if violation_type == "Wrong-side Driving":
            return "CRITICAL" if confidence > 0.6 else "MAJOR"
        if violation_type == "Red-light Violation":
            return "CRITICAL"
        if violation_type == "Stop-line Violation":
            return "MINOR"
        if violation_type == "Seatbelt Non-compliance":
            return "MAJOR"
        if violation_type == "Illegal Parking":
            return "MINOR"
        return "MINOR"

    def compute_composite_confidence(self, vehicle_conf: float, person_confs: list[float]) -> float:
        if not person_confs:
            return float(vehicle_conf)
        avg_person = sum(person_confs) / len(person_confs)
        return float((vehicle_conf * 0.4) + (avg_person * 0.6))

    def is_rider_on_motorcycle(self, person_box: list, moto_box: list, ioa_threshold=0.3) -> bool:
        px1, py1, px2, py2 = person_box
        mx1, my1, mx2, my2 = moto_box
        
        ix1 = max(px1, mx1)
        iy1 = max(py1, my1)
        ix2 = min(px2, mx2)
        iy2 = min(py2, my2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False
            
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        person_area = (px2 - px1) * (py2 - py1)
        
        if person_area == 0:
            return False
            
        ioa = inter_area / person_area
        return ioa > ioa_threshold

    def check_helmet(self, frame, person_box) -> bool:
        px1, py1, px2, py2 = map(int, person_box)
        width = px2 - px1
        height = py2 - py1
        
        head_left = px1 + int(width * 0.25)
        head_right = px2 - int(width * 0.25)
        head_bottom = py1 + int(height * 0.25)
        
        head_crop = frame[py1:head_bottom, head_left:head_right]
        
        if head_crop.size == 0:
            return True

        if self.helmet_classifier is not None:
            try:
                img_tensor = self.helmet_transform(head_crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.helmet_classifier(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    return predicted.item() == 1
            except Exception:
                pass

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        brightness = np.mean(v_channel)
        
        s_channel = hsv[:, :, 1]
        saturation = np.mean(s_channel)
        
        if brightness > 100 and saturation > 40:
            return True
        if brightness < 50:
            return True
            
        return False

    def check_seatbelt(self, frame, person_box) -> bool:
        px1, py1, px2, py2 = map(int, person_box)
        width = px2 - px1
        height = py2 - py1
        
        torso_top = py1 + int(height * 0.2)
        torso_bottom = py1 + int(height * 0.6)
        
        torso_crop = frame[torso_top:torso_bottom, px1:px2]
        
        if torso_crop.size == 0:
            return True
            
        gray = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle = abs(math.degrees(math.atan((y2 - y1) / (x2 - x1))))
                if 30 < angle < 60:
                    return True
        return False

    def check_traffic_light_state(self, frame) -> str:
        h, w = frame.shape[:2]
        top_crop = frame[0:int(h*0.3), 0:w]
        
        if top_crop.size == 0:
            return "GREEN"
            
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
class VideoIngestionEngine:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.detector = ViolationDetector()
        self.ocr_processor = OCRProcessor()

    async def process_image_real(self, image_path: str, stop_line_y: int | None = None, camera_id: str = "cam_01"):
        frame = cv2.imread(image_path)
        if frame is None:
            return []
            
        enhanced_frame, _ = preprocess_frame(frame)
        
        results = self.model.predict(
            frame,
            classes=[0, 2, 3, 5, 7],
            conf=YOLO_CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        persons = []
        cars = []
        motorcycles = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == 0:
                    persons.append(([x1, y1, x2, y2], conf))
                elif cls_id in [2, 5, 7]:
                    cars.append(([x1, y1, x2, y2], conf))
                elif cls_id == 3:
                    motorcycles.append(([x1, y1, x2, y2], conf))

        db = SessionLocal()
        violations_returned = []
        timestamp_dt = datetime.utcnow()

        try:
            for car_box, v_conf in cars:
                vx1, vy1, vx2, vy2 = car_box
                plate_number = self.ocr_processor.extract_license_plate(frame, car_box)

                # Seatbelt
                if not self.detector.check_seatbelt(frame, car_box):
                    severity = self.detector.classify_severity("Seatbelt Non-compliance", 0, v_conf)
                    annotated = enhanced_frame.copy()
                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                    cv2.putText(annotated, f"Seatbelt | {severity}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    filename = f"ev_sb_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                    filepath = os.path.join("uploads", filename)
                    cv2.imwrite(filepath, annotated)

                    v_record = models.ViolationRecord(
                        timestamp=timestamp_dt, camera_id=camera_id, violation_type="Seatbelt Non-compliance", severity=severity,
                        rider_count=0, plate_number=plate_number, confidence=v_conf, image_url=f"/api/images/{filename}"
                    )
                    db.add(v_record)
                    db.flush()
                    violations_returned.append({
                        "id": v_record.id, "camera_id": camera_id, "violation_type": "Seatbelt Non-compliance", "severity": severity,
                        "plate_number": plate_number, "confidence": v_conf, "rider_count": 0, "image_url": v_record.image_url,
                        "timestamp": timestamp_dt.isoformat()
                    })

                # Red-light
                current_tl_state = self.detector.check_traffic_light_state(frame)
                if stop_line_y is not None and vy2 > stop_line_y and current_tl_state == "RED":
                    v_type = "Red-light Violation" if vy2 > stop_line_y + 100 else "Stop-line Violation"
                    severity = self.detector.classify_severity(v_type, 0, v_conf)
                    annotated_sl = enhanced_frame.copy()
                    cv2.rectangle(annotated_sl, (vx1, vy1), (vx2, vy2), (255, 0, 0), 3)
                    cv2.line(annotated_sl, (0, stop_line_y), (annotated_sl.shape[1], stop_line_y), (0, 0, 255), 2)
                    
                    filename = f"ev_sl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                    filepath = os.path.join("uploads", filename)
                    cv2.imwrite(filepath, annotated_sl)

                    v_record = models.ViolationRecord(
                        timestamp=timestamp_dt, camera_id=camera_id, violation_type=v_type, severity=severity,
                        rider_count=0, plate_number=plate_number, confidence=v_conf, image_url=f"/api/images/{filename}"
                    )
                    db.add(v_record)
                    db.flush()
                    violations_returned.append({
                        "id": v_record.id, "camera_id": camera_id, "violation_type": v_type, "severity": severity,
                        "plate_number": plate_number, "confidence": v_conf, "rider_count": 0, "image_url": v_record.image_url,
                        "timestamp": timestamp_dt.isoformat()
                    })

            for moto_box, moto_conf in motorcycles:
                mx1, my1, mx2, my2 = moto_box
                riders_on_moto = []

                for p_box, p_conf in persons:
                    if self.detector.is_rider_on_motorcycle(p_box, moto_box):
                        riders_on_moto.append((p_box, p_conf))
                
                riders_on_moto.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
                riders_on_moto = riders_on_moto[:3]
                rider_count = len(riders_on_moto)
                
                if rider_count >= TRIPLE_RIDING_THRESHOLD:
                    plate_number = self.ocr_processor.extract_license_plate(frame, moto_box)
                    comp_conf = self.detector.compute_composite_confidence(moto_conf, [r[1] for r in riders_on_moto])
                    severity = self.detector.classify_severity("Triple Riding", rider_count, comp_conf)
                    
                    filename = f"ev_tr_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                    filepath = os.path.join("uploads", filename)

                    annotated = enhanced_frame.copy()
                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 0, 255), 3)
                    cv2.imwrite(filepath, annotated)

                    v_record = models.ViolationRecord(
                        timestamp=timestamp_dt, camera_id=camera_id, violation_type="Triple Riding", severity=severity,
                        rider_count=rider_count, plate_number=plate_number, confidence=comp_conf,
                        image_url=f"/api/images/{filename}"
                    )
                    db.add(v_record)
                    db.flush()
                    violations_returned.append({
                        "id": v_record.id, "camera_id": camera_id, "violation_type": "Triple Riding", "severity": severity,
                        "plate_number": plate_number, "confidence": comp_conf, "rider_count": rider_count,
                        "image_url": v_record.image_url, "timestamp": timestamp_dt.isoformat()
                    })

                no_helmet_riders = []
                for r_box, r_conf in riders_on_moto:
                    if not self.detector.check_helmet(frame, r_box):
                        no_helmet_riders.append((r_box, r_conf))

                if no_helmet_riders:
                    plate_number = self.ocr_processor.extract_license_plate(frame, moto_box)
                    comp_conf = max(r[1] for r in no_helmet_riders)
                    severity = self.detector.classify_severity("Helmet Non-compliance", len(no_helmet_riders), comp_conf)
                    
                    filename = f"ev_hl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                    filepath = os.path.join("uploads", filename)
                    annotated = enhanced_frame.copy()
                    cv2.imwrite(filepath, annotated)

                    v_record = models.ViolationRecord(
                        timestamp=timestamp_dt, camera_id=camera_id, violation_type="Helmet Non-compliance", severity=severity,
                        rider_count=len(no_helmet_riders), plate_number=plate_number, confidence=comp_conf,
                        image_url=f"/api/images/{filename}"
                    )
                    db.add(v_record)
                    db.flush()
                    violations_returned.append({
                        "id": v_record.id, "camera_id": camera_id, "violation_type": "Helmet Non-compliance", "severity": severity,
                        "plate_number": plate_number, "confidence": comp_conf, "rider_count": len(no_helmet_riders),
                        "image_url": v_record.image_url, "timestamp": timestamp_dt.isoformat()
                    })

            db.commit()
            return violations_returned
        finally:
            db.close()
    async def process_video_real(self, video_path: str, websocket_manager, stop_line_y: int | None = None, camera_id: str = "cam_01"):
        db = SessionLocal()
        frames_processed = 0
        total_violations = 0
        
        # Initialize DeepSORT Tracker
        tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
        
        # Violation cooldown dictionary: {track_id: {violation_type: last_violation_time}}
        violation_cooldowns = {}
        
        parking_trackers = {}
        vehicle_flow_data = {}
        global_traffic_y_movement = 0.0
        dynamic_flow_direction = "unknown"

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                await websocket_manager.broadcast({"type": "error", "message": f"Cannot open video file: {os.path.basename(video_path)}"})
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            process_every_n = max(1, int(fps / 5)) # Target 5 processing FPS for demo
            
            await websocket_manager.broadcast({
                "type": "status",
                "message": f"Processing started: {total_frames} frames @ {fps:.0f} FPS",
                "total_frames": total_frames
            })

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frames_processed += 1
                if frames_processed % process_every_n != 0:
                    continue

                current_video_time = frames_processed / fps
                timestamp_dt = datetime.utcnow()
                
                enhanced_frame, _ = preprocess_frame(frame)
                
                results = self.model.predict(
                    frame,
                    classes=[0, 2, 3, 5, 7],
                    conf=YOLO_CONFIDENCE_THRESHOLD,
                    verbose=False
                )
                
                persons = []
                bbs_for_tracker = []
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls_id == 0:
                            persons.append(([x1, y1, x2, y2], conf))
                        elif cls_id in [2, 3, 5, 7]:
                            # DeepSort format: ([left, top, w, h], confidence, detection_class)
                            bbs_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))
                            
                # Update Tracker
                tracks = tracker.update_tracks(bbs_for_tracker, frame=frame)
                
                cars = []
                motorcycles = []
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    cls_id = track.det_class
                    
                    if cls_id in [2, 5, 7]:
                        cars.append(([x1, y1, x2, y2], track_id, track.det_conf if track.det_conf is not None else 0.8))
                    elif cls_id == 3:
                        motorcycles.append(([x1, y1, x2, y2], track_id, track.det_conf if track.det_conf is not None else 0.8))

                # Process Cars (Seatbelt, Wrong-side, Stop-line, Parking)
                for car_box, track_id, v_conf in cars:
                    vx1, vy1, vx2, vy2 = car_box
                    cy = (vy1 + vy2) / 2.0
                    
                    if track_id not in violation_cooldowns:
                        violation_cooldowns[track_id] = {}
                    recent = violation_cooldowns[track_id]

                    if track_id not in vehicle_flow_data:
                        vehicle_flow_data[track_id] = {"first_cy": cy, "last_cy": cy}
                    
                    movement_y = cy - vehicle_flow_data[track_id]["first_cy"]
                    global_traffic_y_movement += (cy - vehicle_flow_data[track_id]["last_cy"])
                    vehicle_flow_data[track_id]["last_cy"] = cy

                    if global_traffic_y_movement > 500:
                        dynamic_flow_direction = "down"
                    elif global_traffic_y_movement < -500:
                        dynamic_flow_direction = "up"

                    plate_number = "UNREADABLE"
                    def get_plate():
                        nonlocal plate_number
                        if plate_number == "UNREADABLE":
                            plate_number = self.ocr_processor.extract_license_plate(frame, car_box)
                        return plate_number

                    # Parking (time-based)
                    if vy2 > frame.shape[0] - 50:
                        if track_id not in parking_trackers:
                            parking_trackers[track_id] = current_video_time
                        elif current_video_time - parking_trackers[track_id] > 5.0:
                            if "Illegal Parking" not in recent or current_video_time - recent["Illegal Parking"] > VIOLATION_COOLDOWN_SECONDS:
                                recent["Illegal Parking"] = current_video_time
                                plate = get_plate()
                                filename = f"ev_pk_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                                filepath = os.path.join("uploads", filename)
                                cv2.imwrite(filepath, enhanced_frame)
                                new_record = models.ViolationRecord(
                                    timestamp=timestamp_dt, camera_id=camera_id, violation_type="Illegal Parking",
                                    severity="MINOR", rider_count=0, plate_number=plate, confidence=v_conf, image_url=f"/api/images/{filename}"
                                )
                                db.add(new_record)
                                total_violations += 1
                                await websocket_manager.broadcast({
                                    "camera_id": camera_id, "type": "violation", "violation_type": "Illegal Parking", "severity": "MINOR",
                                    "plate_number": plate, "confidence": v_conf, "rider_count": 0, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                                })
                    else:
                        if track_id in parking_trackers:
                            del parking_trackers[track_id]

                    # Wrong-side driving
                    is_wrong_side = False
                    if dynamic_flow_direction == "down" and movement_y < -50:
                        is_wrong_side = True
                    elif dynamic_flow_direction == "up" and movement_y > 50:
                        is_wrong_side = True

                    if is_wrong_side:
                        if "Wrong-side Driving" not in recent or current_video_time - recent["Wrong-side Driving"] > VIOLATION_COOLDOWN_SECONDS:
                            recent["Wrong-side Driving"] = current_video_time
                            plate = get_plate()
                            severity = self.detector.classify_severity("Wrong-side Driving", 0, v_conf)
                            filename = f"ev_ws_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Wrong-side Driving", severity=severity,
                                rider_count=0, plate_number=plate, confidence=v_conf, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Wrong-side Driving", "severity": severity,
                                "plate_number": plate, "confidence": v_conf, "rider_count": 0, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })

                    # Seatbelt
                    if "Seatbelt Non-compliance" not in recent or current_video_time - recent["Seatbelt Non-compliance"] > VIOLATION_COOLDOWN_SECONDS:
                        if not self.detector.check_seatbelt(frame, car_box):
                            recent["Seatbelt Non-compliance"] = current_video_time
                            plate = get_plate()
                            severity = self.detector.classify_severity("Seatbelt Non-compliance", 0, v_conf)
                            filename = f"ev_sb_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Seatbelt Non-compliance", severity=severity,
                                rider_count=0, plate_number=plate, confidence=v_conf, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Seatbelt Non-compliance", "severity": severity,
                                "plate_number": plate, "confidence": v_conf, "rider_count": 0, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })

                    # Stop-line / Red-light
                    current_tl_state = self.detector.check_traffic_light_state(frame)
                    if stop_line_y is not None and current_tl_state == "RED":
                        if vy2 > stop_line_y:
                            v_type = "Red-light Violation" if vy2 > stop_line_y + 100 else "Stop-line Violation"
                            if v_type not in recent or current_video_time - recent[v_type] > VIOLATION_COOLDOWN_SECONDS:
                                recent[v_type] = current_video_time
                                plate = get_plate()
                                severity = self.detector.classify_severity(v_type, 0, v_conf)
                                filename = f"ev_sl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                                filepath = os.path.join("uploads", filename)
                                cv2.imwrite(filepath, enhanced_frame)
                                new_record = models.ViolationRecord(
                                    timestamp=timestamp_dt, camera_id=camera_id, violation_type=v_type, severity=severity,
                                    rider_count=0, plate_number=plate, confidence=v_conf, image_url=f"/api/images/{filename}"
                                )
                                db.add(new_record)
                                total_violations += 1
                                await websocket_manager.broadcast({
                                    "camera_id": camera_id, "type": "violation", "violation_type": v_type, "severity": severity,
                                    "plate_number": plate, "confidence": v_conf, "rider_count": 0, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                                })

                # Process Motorcycles (Triple Riding, Helmet)
                for moto_box, track_id, moto_conf in motorcycles:
                    if track_id not in violation_cooldowns:
                        violation_cooldowns[track_id] = {}
                    recent = violation_cooldowns[track_id]

                    mx1, my1, mx2, my2 = moto_box
                    riders_on_moto = []
                    for p_box, p_conf in persons:
                        if self.detector.is_rider_on_motorcycle(p_box, moto_box):
                            riders_on_moto.append((p_box, p_conf))
                    
                    riders_on_moto.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
                    riders_on_moto = riders_on_moto[:3]
                    rider_count = len(riders_on_moto)
                    
                    plate_number = "UNREADABLE"
                    def get_plate():
                        nonlocal plate_number
                        if plate_number == "UNREADABLE":
                            plate_number = self.ocr_processor.extract_license_plate(frame, moto_box)
                        return plate_number

                    # Triple Riding
                    if rider_count >= TRIPLE_RIDING_THRESHOLD:
                        if "Triple Riding" not in recent or current_video_time - recent["Triple Riding"] > VIOLATION_COOLDOWN_SECONDS:
                            recent["Triple Riding"] = current_video_time
                            plate = get_plate()
                            comp_conf = self.detector.compute_composite_confidence(moto_conf, [r[1] for r in riders_on_moto])
                            severity = self.detector.classify_severity("Triple Riding", rider_count, comp_conf)
                            filename = f"ev_tr_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Triple Riding", severity=severity,
                                rider_count=rider_count, plate_number=plate, confidence=comp_conf, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Triple Riding", "severity": severity,
                                "plate_number": plate, "confidence": comp_conf, "rider_count": rider_count, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })

                    # Helmet
                    no_helmet_riders = []
                    for r_box, r_conf in riders_on_moto:
                        if not self.detector.check_helmet(frame, r_box):
                            no_helmet_riders.append((r_box, r_conf))

                    if no_helmet_riders:
                        if "Helmet Non-compliance" not in recent or current_video_time - recent["Helmet Non-compliance"] > VIOLATION_COOLDOWN_SECONDS:
                            recent["Helmet Non-compliance"] = current_video_time
                            plate = get_plate()
                            comp_conf = max([r[1] for r in no_helmet_riders]) if no_helmet_riders else moto_conf
                            severity = self.detector.classify_severity("Helmet Non-compliance", len(no_helmet_riders), comp_conf)
                            filename = f"ev_hl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Helmet Non-compliance", severity=severity,
                                rider_count=len(no_helmet_riders), plate_number=plate, confidence=comp_conf, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Helmet Non-compliance", "severity": severity,
                                "plate_number": plate, "confidence": comp_conf, "rider_count": len(no_helmet_riders), "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })

                db.commit()
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
