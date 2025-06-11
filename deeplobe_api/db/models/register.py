from django.db import models

from django.conf import settings

from ..mixins import TimeAuditModel

from django.db.models import JSONField


NEW_CHOICES = [
    ("OCR", "ocr"),
    ("IMAGE_CLASSIFICATION", "image_classification"),
    ("OBJECT_DETECTION", "object_detection"),
    ("IMAGE_SEGMENTATION", "image_segmentation"),
    ("OTHERS", "others"),
]


class Register(TimeAuditModel):
    fullname = models.CharField(max_length=255, null=True, blank=True)
    role = models.CharField(max_length=255, null=True, blank=True)
    type = models.JSONField(default=list, null=True, blank=True)

    class Meta:
        verbose_name = "Register"
        verbose_name_plural = "Registers"
        db_table = "registers"

    def __str__(self):
        return self.fullname
