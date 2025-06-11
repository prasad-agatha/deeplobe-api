from django.db import models

from .workspace import Workspace

from ..mixins import TimeAuditModel

from deeplobe_api.db.models import AIModel, APIKey


MODEL_CHOICES = (
    ("classification", "Custom Classification"),
    ("segmentation", "Custom Segmentation"),
    ("image_similarity", "Custom Image Similarity"),
    ("object_detection", "Custom Object detection"),
    ("ocr", "Custom OCR"),
    ("instance", "Custom Instance"),
    ("pre_sentimental_analysis", "PRE_SENTIMENTAL_ANALYSIS"),
    ("pre_image_similarity", "PRE_IMAGE_SIMILARITY"),
    ("pre_facial_detection", "PRE_FACIAL_DETECTION"),
    ("pre_demographic_recognition", "PRE_DEMOGRAPHIC_RECOGNITION"),
    ("pre_facial_expression", "PRE_FACIAL_EXPRESSION"),
    ("pre_pose_detection", "PRE_POSE_DETECTION"),
    ("pre_text_moderation", "PRE_TEXT_MODERATION"),
    ("pre_people_vehicle_detection", "PRE_PEOPLE_VEHICLE_DETECTION"),
    ("pre_wound_detection", "PRE_WOUND_DETECTION"),
)

class APILogger(TimeAuditModel):
    uuid = models.CharField(max_length=255, unique=True)
    api_key = models.ForeignKey(
        APIKey, related_name="api_loggers", on_delete=models.CASCADE
    )
    workspace = models.ForeignKey(
        Workspace, on_delete=models.CASCADE, null=True, blank=True
    )
    file = models.FileField(
        max_length=255, upload_to="media_file", null=True, blank=True
    )
    model_type = models.CharField(max_length=255, choices=MODEL_CHOICES)
    data = models.JSONField(null=True, blank=True)
    is_custom_model = models.BooleanField(default=False)
    model_id = models.ForeignKey(
        AIModel, related_name="aimodel_id", on_delete=models.CASCADE, null=True
    )
    response_code = models.IntegerField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "APILogger"
        verbose_name_plural = "APILoggers"
        db_table = "apiloggers"
