FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies required for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch first to save ~800MB of download and prevent network timeouts
RUN pip install --no-cache-dir --default-timeout=1000 torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless && pip install opencv-python-headless==4.10.0.84

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
