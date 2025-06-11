from django.db import models

from django.conf import settings

from .workspace import Workspace

from ..mixins import TimeAuditModel


class APICount(TimeAuditModel):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, blank=True, null=True
    )
    subscription_id = models.CharField(max_length=255, blank=True, null=True)
    custom_model_api_count = models.IntegerField()
    workspace = models.ForeignKey(
        Workspace, on_delete=models.CASCADE, null=True, blank=True
    )

    class Meta:
        verbose_name = "APICount"
        verbose_name_plural = "APICounts"
        db_table = "apicounts"
