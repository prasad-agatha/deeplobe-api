from django.conf import settings
from .workspace import Workspace

from django.db import models
from django.db.models import JSONField
from deeplobe_api.db.models import APIKey

from ..mixins import TimeAuditModel


def get_upload_path(instance, filename):
    return f"{instance.model_type}/{instance.uuid}/{filename}"


MODEL_CHOICES = (
    ("classification", "CLASSIFICATION"),
    ("segmentation", "SEGMENTATION"),
    ("image_similarity", "IMAGE_SIMILARITY"),
    ("object_detection", "OBJECT_DETECTION"),
    ("ocr", "OCR"),
    ("instance", "INSTANCE"),
    ("llm","LLM")
)


class AIModel(TimeAuditModel):
    """[summary]
    Args:
        TimeAuditModel ([type]): [description]
    """

    model_type = models.CharField(max_length=255, choices=MODEL_CHOICES)
    # task uuid
    uuid = models.CharField(max_length=255, unique=True)

    # name
    weight_file = models.TextField(null=True, blank=True)

    aws_weight_file = models.URLField(null=True, blank=True)

    annotation_file = models.FileField(upload_to=get_upload_path, null=True, blank=True)

    key_file = models.URLField(null=True, blank=True)

    # user
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    workspace = models.ForeignKey(
        Workspace, on_delete=models.CASCADE, null=True, blank=True
    )

    api_key = models.CharField(max_length=256, null=True, blank=True)

    status = models.CharField(max_length=256, default="Draft")

    # data
    data = JSONField(blank=True, null=True)

    # extra
    extra = JSONField(blank=True, null=True)

    description = models.CharField(max_length=255, blank=True)

    name = models.CharField(max_length=255, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    is_trained = models.BooleanField(default=False)

    is_failed = models.BooleanField(default=False)

    is_active = models.BooleanField(default=True)

    last_used = models.DateTimeField(null=True, blank=True)

    job = JSONField(blank=True, null=True)

    failed_reason = models.JSONField(default=dict)

    preprocessing = JSONField(blank=True, null=True)

    augmentation = JSONField(blank=True, null=True)

    class Meta:
        verbose_name = "AIModel"
        verbose_name_plural = "AIModels"
        db_table = "aimodels"
        ordering = ["-created"]

    def __str__(self):
        return f"{self.user.email}/ {self.uuid}"


class AIModelPrediction(TimeAuditModel):
    aimodel = models.ForeignKey(AIModel, max_length=255, on_delete=models.CASCADE)

    input_images = models.JSONField()

    result = models.JSONField(null=True)

    # user
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    # data
    data = JSONField(blank=True, null=True)

    # extra
    extra = JSONField(blank=True, null=True)

    description = models.CharField(max_length=255, null=True, blank=True)

    created = models.DateTimeField(auto_now_add=True)

    updated = models.DateTimeField(auto_now=True)

    is_predicted = models.BooleanField(default=False)

    is_failed = models.BooleanField(default=False)

    class Meta:
        verbose_name = "AIModelPrediction"
        verbose_name_plural = "AIModelPredictions"
        db_table = "aimodelpredictions"
        ordering = ["-created"]

    def __str__(self):
        return f"{self.user.email}"
