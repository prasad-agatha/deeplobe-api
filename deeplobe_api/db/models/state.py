from django.db import models

from ..mixins import TimeAuditModel


class State(TimeAuditModel):

    name = models.CharField(max_length=255, null=True, blank=True)
    is_country = models.BooleanField(default=False)
    parent_name = models.CharField(max_length=255, null=True, blank=True)
    currency = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        verbose_name = "State"
        verbose_name_plural = "States"
        db_table = "states"

    def __str__(self):
        return f"{self.name}"
