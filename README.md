# 🚨 ViolationVision AI

![ViolationVision Platform](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![YOLOv8](https://img.shields.io/badge/YOLOv8-FF0000?style=for-the-badge&logo=yolo)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-0088CC?style=for-the-badge)

**ViolationVision AI** is a state-of-the-art Traffic Intelligence Platform designed to automatically detect and log traffic violations (such as Triple Riding) from CCTV video feeds using Computer Vision and Machine Learning. 

Built for high-performance and real-time processing, the platform features a highly responsive frontend dashboard to view live anomalies, monitor violation confidence levels, and export evidence directly to CSV.

---

## 🌟 Key Features

*   **Real-Time Video Ingestion**: Process `.mp4` CCTV footage instantly with high performance.
*   **Triple Riding Detection**: Custom Spatial AI using Ultralytics YOLOv8 to accurately map riders to motorcycles and count riders, avoiding background pedestrians.
*   **Intelligent Object Tracking & NMS**: Deep Non-Maximum Suppression and temporal tracking to ensure the exact same vehicle isn't flagged multiple times within a 60-second cooldown window.
*   **License Plate Extraction**: Integrated with PaddleOCR to crop, scan, and read the license plate of violating vehicles on the fly.
*   **Live Dashboard**: A stunning React + TailwindCSS dashboard connected via WebSockets to instantly stream violations directly to the operator's screen as they happen.
*   **Evidence Generation**: Automatically crops and annotates bounding boxes on the raw frame and saves them as visual proof.
*   **Persistent Logging & Export**: Built on SQLAlchemy/SQLite to permanently log violation data with an easy-to-use **CSV Export** button.

---

## 🏗️ Architecture

1.  **Frontend (React + Vite + TailwindCSS)**: 
    *   Dark-mode, premium UI/UX designed for rapid visual assessment.
    *   WebSocket listener (`/ws/alerts`) for live updates.
2.  **Backend (FastAPI + Uvicorn)**:
    *   High-concurrency async endpoints.
    *   `ml_pipeline.py`: Handles chunked video streaming, bounding box geometric mapping, and OCR extraction in a background thread to prevent blocking the event loop.
3.  **Machine Learning**:
    *   `yolov8n.pt`: Real-time object detection (Classes 0 for Person, 3 for Motorcycle).
    *   `PaddleOCR`: Text detection and recognition (English/Numeric).

---

## 🚀 Quick Start Guide

### Prerequisites
*   Docker & Docker Compose installed on your machine.
*   Git

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/violation-vision-mvp.git
cd violation-vision-mvp
```

### 2. Launch the Platform
Bring up both the API and the UI using Docker Compose. The configuration is already optimized for deployment.
```bash
docker-compose up -d --build
```

*   **Frontend**: Available at `http://localhost:3000`
*   **Backend API**: Available at `http://localhost:8000`

### 3. Usage
1. Open `http://localhost:3000` in your browser.
2. Click on the **Video Ingestion Engine** box and upload a CCTV `.mp4` file.
3. Watch the dashboard instantly populate with live violations as the AI processes the frames.
4. Click **📥 Export CSV** at the top right to download a full report of all recorded evidence.

---

## 🛠️ Configuration
You can tweak the AI strictness in `backend/config.py` (or via environment variables in `docker-compose.yml`):
*   `YOLO_CONF_THRESHOLD`: Controls the YOLO bounding box confidence (Default: 0.25)
*   `TRIPLE_RIDING_THRESHOLD`: Minimum riders mapped to a motorcycle to trigger an alert (Default: 3)
*   `VIOLATION_COOLDOWN`: Cooldown in seconds before the same vehicle gets flagged again (Default: 60.0s)

---

## 🏆 Hackathon Highlights
What makes ViolationVision stand out?
*   **Zero-Hallucination Geometry**: Rather than just blindly counting people and bikes, our pipeline features a strict custom geometric algorithm (`is_rider_on_motorcycle`) that uses intersection-over-union, spatial margins, and size comparisons to mathematically prove a person is *actually riding* the bike, not just standing behind it.
*   **Production-Ready Concurrency**: The Heavy ML processing is safely detached into an `asyncio` ThreadPoolExecutor to prevent blocking the FastAPI WebSocket loop.
*   **Dockerized with Precision**: Resolved deep glibc/C++ conflicts between OpenCV and PaddleOCR by carefully pinning dependencies to `debian:bullseye` and stripping redundant `headless` packages.
*   **Flawless UX**: No page reloads needed. The live dashboard feels like a real-time command center.

---

Made with ❤️ for the Flipkart GRiD Hackathon.
