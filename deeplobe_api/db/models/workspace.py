from django.db import models

from ..mixins import TimeAuditModel

from deeplobe_api.db.models import User


class Workspace(TimeAuditModel):
    name = models.CharField(max_length=256)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Workspace"
        verbose_name_plural = "Workspaces"
        db_table = "workspace"

    def __str__(self):
        return self.name
