import os
import asyncio
from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery = Celery(
    "violation_vision",
    broker=broker_url,
    backend=broker_url
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # A single worker processes one video at a time to prevent GPU OOM
    worker_concurrency=1,
)

@celery.task(name="process_video_task")
def process_video_task(video_path: str, stop_line_y: int | None, camera_id: str):
    """
    Celery task to process a video file using the VideoIngestionEngine.
    We need to run the async engine function within a synchronous Celery task.
    """
    import sys
    import os
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
        
    from ml_pipeline import VideoIngestionEngine
    import json
    import redis

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
    r = redis.Redis.from_url(redis_url)

    class RedisWebsocketManagerMock:
        """
        Mocks the FastAPI WebSocket manager by publishing events to Redis Pub/Sub,
        which the FastAPI server will subscribe to and broadcast to connected clients.
        """
        async def broadcast(self, message: dict):
            # Publish synchronously to Redis
            r.publish("ws_events", json.dumps(message))

    mock_manager = RedisWebsocketManagerMock()
    
    # Initialize Engine (heavy ML models load here if not already loaded in worker)
    engine = VideoIngestionEngine()
    
    # Run the async pipeline loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        engine.process_video_real(video_path, mock_manager, stop_line_y, camera_id)
    )

    return {"status": "success", "video": video_path}
