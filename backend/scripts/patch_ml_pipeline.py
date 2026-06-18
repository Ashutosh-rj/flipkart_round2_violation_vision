import re
import os

def main():
    file_path = r"d:\Hackthaon\flipkart_round2\violation-vision-mvp\backend\ml_pipeline.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add import re
    if "import re" not in content:
        content = content.replace("import traceback", "import traceback\nimport re")

    # 2. Update OCR extract_license_plate
    old_ocr = """            texts = [line[1][0] for line in results[0]]
            plate_text = "".join(texts).replace(" ", "").upper()
            
            if len(plate_text) >= 4:
                return plate_text"""
    
    new_ocr = """            texts = [line[1][0] for line in results[0]]
            plate_text = "".join(texts).replace(" ", "").upper()
            plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)
            
            if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$', plate_text):
                return plate_text
            elif len(plate_text) >= 4 and re.search(r'[A-Z]{2}.*[0-9]{4}', plate_text):
                return plate_text"""
    content = content.replace(old_ocr, new_ocr)

    # 3. Remove apply_nms
    nms_pattern = re.compile(r'    def apply_nms\(self, boxes: list, scores: list, iou_threshold: float = 0\.45\) -> list\[int\]:.*?(?=    def is_rider_on_motorcycle)', re.DOTALL)
    content = re.sub(nms_pattern, '', content)

    # 4. In process_video_real, add flow and parking tracking dicts
    old_init = """        # Violation cooldown dictionary: {track_id: {violation_type: last_violation_time}}
        violation_cooldowns = {}

        try:"""
    new_init = """        # Violation cooldown dictionary: {track_id: {violation_type: last_violation_time}}
        violation_cooldowns = {}
        
        parking_trackers = {}
        vehicle_flow_data = {}
        global_traffic_y_movement = 0.0
        dynamic_flow_direction = "unknown"

        try:"""
    content = content.replace(old_init, new_init)

    # 5. Fix loop extracting cars and motorcycles
    old_append = """                    if cls_id in [2, 5, 7]:
                        cars.append(([x1, y1, x2, y2], track_id))
                    elif cls_id == 3:
                        motorcycles.append(([x1, y1, x2, y2], track_id))"""
    new_append = """                    if cls_id in [2, 5, 7]:
                        cars.append(([x1, y1, x2, y2], track_id, track.det_conf if track.det_conf is not None else 0.8))
                    elif cls_id == 3:
                        motorcycles.append(([x1, y1, x2, y2], track_id, track.det_conf if track.det_conf is not None else 0.8))"""
    content = content.replace(old_append, new_append)

    # 6. Process Cars
    old_cars = """                # Process Cars (Seatbelt, Wrong-side, Stop-line, Parking)
                for car_box, track_id in cars:
                    vx1, vy1, vx2, vy2 = car_box
                    # Mock flow map for wrong-side logic
                    flow_map = np.zeros((10, 10))
                    
                    if track_id not in violation_cooldowns:
                        violation_cooldowns[track_id] = {}
                    recent = violation_cooldowns[track_id]

                    plate_number = "UNREADABLE"
                    def get_plate():
                        nonlocal plate_number
                        if plate_number == "UNREADABLE":
                            plate_number = self.ocr_processor.extract_license_plate(frame, car_box)
                        return plate_number

                    # Parking (mock duration)
                    if "Illegal Parking" not in recent or current_video_time - recent["Illegal Parking"] > VIOLATION_COOLDOWN_SECONDS:
                        if vy2 > frame.shape[0] - 50: # Mock stationary
                            recent["Illegal Parking"] = current_video_time
                            plate = get_plate()
                            filename = f"ev_pk_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Illegal Parking",
                                severity="MINOR", rider_count=0, plate_number=plate, confidence=1.0, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Illegal Parking", "severity": "MINOR",
                                "plate_number": plate, "confidence": 1.0, "rider_count": 0, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })

                    # Seatbelt
                    if "Seatbelt Non-compliance" not in recent or current_video_time - recent["Seatbelt Non-compliance"] > VIOLATION_COOLDOWN_SECONDS:
                        if not self.detector.check_seatbelt(frame, car_box):
                            recent["Seatbelt Non-compliance"] = current_video_time
                            plate = get_plate()
                            severity = self.detector.classify_severity("Seatbelt Non-compliance", 0, 1.0)
                            filename = f"ev_sb_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Seatbelt Non-compliance", severity=severity,
                                rider_count=0, plate_number=plate, confidence=1.0, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Seatbelt Non-compliance", "severity": severity,
                                "plate_number": plate, "confidence": 1.0, "rider_count": 0, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })

                    # Stop-line / Red-light
                    current_tl_state = self.detector.check_traffic_light_state(frame)
                    if stop_line_y is not None and current_tl_state == "RED":
                        if vy2 > stop_line_y:
                            v_type = "Red-light Violation" if vy2 > stop_line_y + 100 else "Stop-line Violation"
                            if v_type not in recent or current_video_time - recent[v_type] > VIOLATION_COOLDOWN_SECONDS:
                                recent[v_type] = current_video_time
                                plate = get_plate()
                                severity = self.detector.classify_severity(v_type, 0, 1.0)
                                filename = f"ev_sl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                                filepath = os.path.join("uploads", filename)
                                cv2.imwrite(filepath, enhanced_frame)
                                new_record = models.ViolationRecord(
                                    timestamp=timestamp_dt, camera_id=camera_id, violation_type=v_type, severity=severity,
                                    rider_count=0, plate_number=plate, confidence=1.0, image_url=f"/api/images/{filename}"
                                )
                                db.add(new_record)
                                total_violations += 1
                                await websocket_manager.broadcast({
                                    "camera_id": camera_id, "type": "violation", "violation_type": v_type, "severity": severity,
                                    "plate_number": plate, "confidence": 1.0, "rider_count": 0, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                                })"""

    new_cars = """                # Process Cars (Seatbelt, Wrong-side, Stop-line, Parking)
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
                                })"""
    content = content.replace(old_cars, new_cars)

    # 7. Process Motorcycles
    old_moto = """                # Process Motorcycles (Triple Riding, Helmet)
                for moto_box, track_id in motorcycles:"""
    new_moto = """                # Process Motorcycles (Triple Riding, Helmet)
                for moto_box, track_id, moto_conf in motorcycles:"""
    content = content.replace(old_moto, new_moto)

    old_tr = """                            severity = self.detector.classify_severity("Triple Riding", rider_count, 1.0)
                            filename = f"ev_tr_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Triple Riding", severity=severity,
                                rider_count=rider_count, plate_number=plate, confidence=1.0, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Triple Riding", "severity": severity,
                                "plate_number": plate, "confidence": 1.0, "rider_count": rider_count, "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })"""
    new_tr = """                            comp_conf = self.detector.compute_composite_confidence(moto_conf, [r[1] for r in riders_on_moto])
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
                            })"""
    content = content.replace(old_tr, new_tr)

    old_helmet = """                            severity = self.detector.classify_severity("Helmet Non-compliance", len(no_helmet_riders), 1.0)
                            filename = f"ev_hl_{timestamp_dt.strftime('%Y%m%d%H%M%S%f')}.jpg"
                            filepath = os.path.join("uploads", filename)
                            cv2.imwrite(filepath, enhanced_frame)
                            new_record = models.ViolationRecord(
                                timestamp=timestamp_dt, camera_id=camera_id, violation_type="Helmet Non-compliance", severity=severity,
                                rider_count=len(no_helmet_riders), plate_number=plate, confidence=1.0, image_url=f"/api/images/{filename}"
                            )
                            db.add(new_record)
                            total_violations += 1
                            await websocket_manager.broadcast({
                                "camera_id": camera_id, "type": "violation", "violation_type": "Helmet Non-compliance", "severity": severity,
                                "plate_number": plate, "confidence": 1.0, "rider_count": len(no_helmet_riders), "image_url": f"/api/images/{filename}", "timestamp": timestamp_dt.isoformat()
                            })"""
    new_helmet = """                            comp_conf = max([r[1] for r in no_helmet_riders]) if no_helmet_riders else moto_conf
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
                            })"""
    content = content.replace(old_helmet, new_helmet)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        print("Updated ml_pipeline.py successfully!")

if __name__ == "__main__":
    main()
