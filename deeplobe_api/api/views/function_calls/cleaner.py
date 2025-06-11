import os
import shutil
import calendar
import datetime
import django_rq

from django_rq.queues import get_queue

from deeplobe_api.db.models.aimodel import AIModel


redis_conn = django_rq.get_connection("default")
queue = get_queue("default")


def clean_aimodel_local_folder(repeat=False):
    # Inactive models filtering
    aimodels = AIModel.objects.filter(is_active=False)
    for aimodel in aimodels:
        # Removing inactive models in local path including ckpt
        local_path = f"data/{aimodel.uuid}/"
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
    # Removing all inactive model objects
    # aimodels.delete()
    if repeat:
        today = datetime.date.today()
        next_saturday = today + datetime.timedelta(
            (calendar.SATURDAY - today.weekday()) % 7
        )
        queue.enqueue_at(
            datetime.datetime(
                next_saturday.year, next_saturday.month, next_saturday.day, 0, 0
            ),
            clean_aimodel_local_folder,
            True,
        )
