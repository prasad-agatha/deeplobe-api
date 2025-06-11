from django.db import models

from django.conf import settings

from ..mixins import TimeAuditModel

from django.db.models import JSONField


class Task(TimeAuditModel):
    """[summary]
    Args:
        TimeAuditModel ([type]): [description]
    """

    # task uuid
    uuid = models.CharField(max_length=255)

    # name
    weight_name = models.CharField(max_length=255, blank=True)

    # kind of task
    task_type = models.CharField(max_length=255, blank=True)

    # user
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    ckpt_file_path = models.FileField(upload_to="model_memory", null=True, blank=True)

    # finished task
    task_finished = models.BooleanField(default=False)

    # data
    data = JSONField(blank=True, null=True)

    # extra
    extra = JSONField(blank=True, null=True)

    description = models.CharField(max_length=255, blank=True)

    name = models.CharField(max_length=255, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    is_failed = models.BooleanField(default=False)

    is_successful = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Task"
        verbose_name_plural = "Tasks"
        db_table = "tasks"
        ordering = ["-created"]

    def __str__(self):
        return f"{self.user.email}/ {self.weight_name}/ {self.uuid}"
