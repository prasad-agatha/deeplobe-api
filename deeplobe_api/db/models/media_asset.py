from django.db import models

from ..mixins import TimeAuditModel


class MediaAsset(TimeAuditModel):
    name = models.CharField(max_length=255)
    asset = models.FileField(max_length=255, upload_to="media_assets")
    class_name = models.TextField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name
