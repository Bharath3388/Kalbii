from __future__ import annotations
from celery import Celery
from app.core.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "kalbii",
    broker=_settings.REDIS_URL,
    backend=_settings.REDIS_URL,
    include=["app.workers.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_max_tasks_per_child=50,
    broker_connection_retry_on_startup=True,
)
