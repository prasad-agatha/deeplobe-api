from .base import BaseModelSerializer

from deeplobe_api.db.models import Task


class TaskSerializer(BaseModelSerializer):
    class Meta:
        model = Task
        fields = [
            "id",
            "uuid",
            "weight_name",
            "task_type",
            "user",
            "task_finished",
            "data",
            "extra",
            "name",
            "description",
            "created",
            "updated",
        ]
