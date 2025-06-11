from django.db import models

from ..mixins import TimeAuditModel

from deeplobe_api.db.models import User

STATUS_CHOICES = (
    ("new", "NEW"),
    ("withdrawn", "WITHDRAWN"),
    ("assigned", "ASSIGNED"),
    ("pending", "PENDING"),
    ("resolved", "RESOLVED"),
    ("open", "OPEN"),
    ("closed", "CLOSED"),
    ("active", "ACTIVE"),
    ("inactive", "INACTIVE"),
    ("contacted", "CONTACTED"),
    ("uncontacted", "UNCONTACTED"),
)


class ContactUs(TimeAuditModel):
    """[summary]
    Args:
        TimeAuditModel ([type]): [description]
    """

    name = models.CharField(max_length=255, null=True, blank=True)

    contact_number = models.CharField(max_length=255, null=True, blank=True)

    description = models.TextField(null=True, blank=True)

    hearing = models.TextField(null=True, blank=True)

    email = models.CharField(max_length=255, null=True, blank=True)

    role = models.CharField(max_length=255, null=True, blank=True)

    company = models.CharField(max_length=255, blank=True, null=True)

    title = models.CharField(max_length=255, blank=True, null=True)

    notes = models.TextField(null=True, blank=True)

    requested_by = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )

    status = models.CharField(max_length=255, default="new", choices=STATUS_CHOICES)

    class Meta:
        verbose_name = "Contact Us"
        verbose_name_plural = "Contact Us"
        db_table = "contact_us"

    def __str__(self):
        return f"{self.name}"
