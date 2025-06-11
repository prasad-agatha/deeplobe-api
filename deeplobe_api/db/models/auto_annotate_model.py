from django.db import models

from django.conf import settings

from ..mixins import TimeAuditModel

from django.db.models import JSONField


class AutoAnnotatePredictionModel(TimeAuditModel):

    input_images = models.JSONField()

    result = models.JSONField(null=True)

    # user
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    # extra
    extra = JSONField(blank=True, null=True)

    class Meta:
        verbose_name = "AutoAnnotatePredictionModel"
        verbose_name_plural = " AutoAnnotatePredictionModels"
        db_table = "autoannotatepredictionmodel"

    def __str__(self):
        return f"{self.user.email}"
