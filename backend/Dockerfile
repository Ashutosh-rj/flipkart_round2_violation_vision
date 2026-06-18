FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch first to save ~800MB of download and prevent network timeouts
RUN pip install --no-cache-dir --default-timeout=1000 torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
