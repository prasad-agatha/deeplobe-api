from decouple import config

from django.db import models

from datetime import timedelta

from django.utils import timezone

from ..mixins import TimeAuditModel

from deeplobe_api.db.models import User


class APIKey(TimeAuditModel):
    user = models.ForeignKey(User, related_name="api_keys", on_delete=models.CASCADE)
    key = models.CharField(max_length=256)
    secret = models.CharField(max_length=256, null=True, blank=True)
    expire_date = models.DateField(null=True, blank=True)
    active = models.BooleanField(default=True)
    aimodel = models.CharField(max_length=256, null=True, blank=True)
    pretrained_model = models.CharField(max_length=256, null=True, blank=True)
    application_name = models.CharField(max_length=256, null=True, blank=True)

    class Meta:
        verbose_name = "APIKey"
        verbose_name_plural = "APIKeys"
        db_table = "api_keys"
        unique_together = (("user", "aimodel"), ("user", "pretrained_model"))

    def __str__(self):
        return self.user.username

    # def save(self, *args, **kwargs):
    #     days = config("API_KEY_EXPIRY_DAYS", cast=int)
    #     expire_date = (timezone.now() + timedelta(days=days)).date()
    #     self.expire_date = expire_date
    #     super(APIKey, self).save(*args, **kwargs)
