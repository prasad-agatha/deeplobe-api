from django.db import models

from django.conf import settings

from ..mixins import TimeAuditModel


def get_upload_path(instance, filename):
    return f"{instance.model_name}/{instance.uuid}/{instance.category_name}/{filename}"


class FileAssets(TimeAuditModel):
    model_name = models.CharField(max_length=255)
    uuid = models.CharField(max_length=255)
    photo = models.FileField(upload_to=get_upload_path)
    category_name = models.CharField(max_length=255)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
