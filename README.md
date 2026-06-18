# 🚨 ViolationVision AI

![ViolationVision Platform](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![YOLOv8](https://img.shields.io/badge/YOLOv8-FF0000?style=for-the-badge&logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)

**ViolationVision AI** is a state-of-the-art Traffic Intelligence Platform designed to automatically detect and log traffic violations from CCTV video feeds using Computer Vision and Machine Learning. 

Built for high-performance and real-time processing, the platform features a highly responsive frontend dashboard to view live anomalies, monitor violation confidence levels, and export evidence directly to CSV.

---

## 🌟 Key Features

*   **Real-Time Video Ingestion**: Process `.mp4` CCTV footage instantly with high performance.
*   **7 Integrated Violation Detectors**: 
    1. Triple Riding 
    2. Helmet Non-compliance
    3. Seatbelt Non-compliance
    4. Wrong-side Driving
    5. Red-light Violation
    6. Stop-line Violation
    7. Illegal Parking
*   **Auto-Adaptive Intelligence**: Features dynamic auto-gamma correction for extreme lighting/weather, and crowd-sourced vehicle flow tracking to automatically learn the correct driving direction of any road without manual configuration.
*   **Intelligent Object Tracking**: Deep tracking to ensure the exact same vehicle isn't flagged multiple times within a 30-second cooldown window (tracked per-vehicle).
*   **License Plate Extraction**: Integrated to crop, scan, and extract plates using OCR techniques.
*   **Live Dashboard**: A stunning React + TailwindCSS dashboard connected via WebSockets to instantly stream violations directly to the operator's screen as they happen.
*   **Evidence Generation**: Automatically crops and annotates bounding boxes on the raw frame and saves them as visual proof.
*   **Persistent Logging**: Built on SQLAlchemy/SQLite to permanently log violation data with an easy-to-use **CSV Export** button.

---

## 🏗️ Architecture

1.  **Frontend (React + Vite + TailwindCSS)**: 
    *   Dark-mode, premium UI/UX designed for rapid visual assessment.
    *   WebSocket listener (`/ws/alerts`) for live updates.
2.  **Backend (FastAPI + Uvicorn)**:
    *   High-concurrency async endpoints.
    *   `ml_pipeline.py`: Handles chunked video streaming, bounding box geometric mapping, and heuristics in a background thread to prevent blocking the event loop.
3.  **Machine Learning**:
    *   `yolov8n.pt`: Real-time object detection (Classes 0 for Person, 2 for Car, 3 for Motorcycle, 5 for Bus, 7 for Truck).

---

## 🚀 Quick Start Guide

### Prerequisites
*   Docker & Docker Compose installed on your machine.

### 1. Launch the Platform
Bring up both the API and the UI using Docker Compose. The configuration is already optimized for deployment.
```bash
docker-compose up -d --build
```

*   **Frontend**: Available at `http://localhost:3000`
*   **Backend API**: Available at `http://localhost:8000`

### 2. Usage
1. Open `http://localhost:3000` in your browser.
2. Click on the **Video Ingestion Engine** box and upload a CCTV `.mp4` file.
3. Watch the dashboard instantly populate with live violations as the AI processes the frames.
4. Click **📥 Export CSV** at the top right to download a full report of all recorded evidence.

---

## 🛠️ Configuration
You can tweak the AI strictness in `backend/config.py`:
*   `YOLO_CONF_THRESHOLD`: Controls the YOLO bounding box confidence (Default: 0.20)
*   `TRIPLE_RIDING_THRESHOLD`: Minimum riders mapped to a motorcycle to trigger an alert (Default: 3)
*   `VIOLATION_COOLDOWN_SECONDS`: Cooldown before the same vehicle gets flagged again (Default: 30.0s)

---

## 🏆 Hackathon Highlights
What makes ViolationVision stand out?
*   **Zero-Configuration Deployment**: Thanks to our dynamic traffic flow algorithm, the system automatically learns which way traffic should flow based on the global tally of all vehicles, meaning it can be dropped into any CCTV camera feed in the world without manually telling it which way is "wrong-way".
*   **CCTV Top-Down Adaptability**: We use robust Intersection-over-Area (IoA) maths to map riders to motorcycles, making the AI immune to the severe isometric distortion caused by high-mounted traffic cameras.
*   **Production-Ready Concurrency**: The Heavy ML processing is safely detached into an `asyncio` ThreadPoolExecutor to prevent blocking the FastAPI WebSocket loop.
*   **Flawless UX**: No page reloads needed. The live dashboard feels like a real-time command center.

---

Made with ❤️ for the Flipkart GRiD Hackathon.
