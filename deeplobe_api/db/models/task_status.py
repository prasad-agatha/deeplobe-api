from .task import Task

from django.db import models

from ..mixins import TimeAuditModel

from django.db.models import JSONField


class TaskStatus(TimeAuditModel):
    """[summary]
    Args:
        TimeAuditModel ([type]): [description]
    """

    # kind of task
    process_type = models.CharField(max_length=255, blank=True)

    # train status
    process_status = models.BooleanField(default=False)

    task = models.ForeignKey(Task, on_delete=models.CASCADE)

    # data
    data = JSONField(blank=True, null=True)

    # extra
    extra = JSONField(blank=True, null=True)

    class Meta:
        verbose_name = "Task Status"
        verbose_name_plural = "Task Status"
        db_table = "task_status"

    def __str__(self):
        return f"{self.process_type}/ {self.task}"
