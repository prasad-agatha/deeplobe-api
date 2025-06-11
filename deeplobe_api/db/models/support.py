from django.db import models

from django.conf import settings

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


class Support(TimeAuditModel):
    requested_by = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    name = models.CharField(max_length=255, null=True, blank=True)
    email = models.CharField(max_length=255, null=True, blank=True)
    subject = models.CharField(max_length=255, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    file = models.FileField(upload_to="media_file", null=True, blank=True)

    class Meta:
        verbose_name = "Support"
        verbose_name_plural = "Supports"
        db_table = "supports"
        ordering = ("created",)

    def __str__(self):
        return self.subject


class SupportTicket(TimeAuditModel):
    ticket_id = models.ForeignKey(
        Support, on_delete=models.CASCADE, null=True, blank=True
    )
    status = models.CharField(max_length=255, default="new", choices=STATUS_CHOICES)

    class Meta:
        verbose_name = "SupportTicket"
        verbose_name_plural = "SupportTickets"
        db_table = "support_tickets"

    # def __str__(self):
    #     return self
