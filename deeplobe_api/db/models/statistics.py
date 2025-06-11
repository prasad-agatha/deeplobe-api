from django.db import models

from ..mixins import TimeAuditModel

from deeplobe_api.db.models.aimodel import AIModel


class Statistic(TimeAuditModel):
    aimodel_id = models.ForeignKey(
        AIModel, related_name="model_id", on_delete=models.CASCADE
    )
    data = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Statistic"
        verbose_name_plural = "Statistics"
        db_table = "statistics"

    def __str__(self):
        return str(self.aimodel_id.id)
