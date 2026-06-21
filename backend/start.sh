#!/bin/bash
# Start Celery worker in the background
celery -A celery_app.celery worker --loglevel=info -c 1 &

# Start Uvicorn in the foreground (Hugging Face expects port 7860)
uvicorn main:app --host 0.0.0.0 --port 7860
