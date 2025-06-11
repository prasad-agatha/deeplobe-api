from django.db import models

from django.conf import settings

from ..mixins import TimeAuditModel

from django.db.models import JSONField


class SocialProvider(TimeAuditModel):
    """
    Model definition for SocialProvider.
    """

    SOCIAL_PROVIDERS = ("GOOGLE", "Google")

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, blank=True, null=True
    )

    provider = models.CharField(max_length=255)

    extra = JSONField(null=True)

    class Meta:
        """Meta definition for SocialProvider."""

        db_table = "social_providers"
        verbose_name = "Social Provider"
        verbose_name_plural = "Social Providers"

    def __str__(self):
        """Unicode representation of SocialProvider."""
        return self.user.username
