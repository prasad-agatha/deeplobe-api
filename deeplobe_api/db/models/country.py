from django.db import models

from ..mixins import TimeAuditModel


class Country(TimeAuditModel):

    name = models.CharField(max_length=255, null=True, blank=True)
    is_country = models.BooleanField(default=False)
    parent_location = models.CharField(max_length=255, null=True, blank=True)
    currency = models.CharField(max_length=255, null=True, blank=True)
    currency_symbol = models.CharField(max_length=255, null=True, blank=True)
    iso = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        verbose_name = "Country"
        verbose_name_plural = "Countries"
        db_table = "countries"

    def __str__(self):
        return f"{self.name}"
