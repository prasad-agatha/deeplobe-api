from .base import BaseModelSerializer

from deeplobe_api.db.models import TaskStatus


class TaskStatusSerializer(BaseModelSerializer):
    class Meta:
        model = TaskStatus
        fields = ["id", "process_type", "process_status", "task", "data", "extra"]
