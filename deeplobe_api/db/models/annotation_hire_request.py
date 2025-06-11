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


class AnnotationHireExpertRequest(TimeAuditModel):
    """[summary]
    Args:
        TimeAuditModel ([type]): [description]
    """

    requested_by = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    name = models.CharField(max_length=255, null=True, blank=True)

    email = models.CharField(max_length=255, null=True, blank=True)

    subject = models.CharField(max_length=255, null=True, blank=True)

    description = models.TextField(null=True, blank=True)

    notes = models.TextField(null=True, blank=True)

    requested_by = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )

    type = models.CharField(max_length=255, default="annotation")

    status = models.CharField(max_length=255, default="new", choices=STATUS_CHOICES)

    class Meta:
        verbose_name = "Annotation Hire Expert Request"
        verbose_name_plural = "Annotation Hire Expert Requests"
        db_table = "annotation_hire_expert_requests"

    def __str__(self):
        return f"{self.subject}"
