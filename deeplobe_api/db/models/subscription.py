from django.db import models

from .workspace import Workspace

from ..mixins import TimeAuditModel

from deeplobe_api.db.models import User


ROLE_CHOICES = (
    ("collaborator", "collaborator"),
    ("annotator", "annotator"),
)


class Subscription(TimeAuditModel):
    customer_id = models.CharField(max_length=256, null=True, blank=True)
    subscription_id = models.CharField(max_length=255, blank=True, null=True)
    subscriber = models.OneToOneField(User, on_delete=models.CASCADE)
    subscription_plan = models.CharField(
        max_length=255, default="DeepLobe-Free-Plan-USD-Monthly"
    )

    class Meta:
        verbose_name = "Subscription"
        verbose_name_plural = "Subscriptions"
        db_table = "subscriptions"

    def __str__(self):
        return self.subscriber.username


class UserInvitation(TimeAuditModel):
    subscription = models.CharField(max_length=256, null=True, blank=True)
    invitee = models.ForeignKey(
        User, related_name="user_invitee", on_delete=models.CASCADE
    )
    workspace = models.ForeignKey(
        Workspace, on_delete=models.CASCADE, null=True, blank=True
    )
    collaborator_email = models.CharField(max_length=256, null=True, blank=True)
    is_collaborator = models.BooleanField(default=False)
    role = models.CharField(max_length=255, choices=ROLE_CHOICES, null=True, blank=True)
    models = models.JSONField(default=list, null=True, blank=True)

    class Meta:
        verbose_name = "UserInvitation"
        verbose_name_plural = "User Invitations"
        db_table = "user_invitations"

    def __str__(self):
        return self.invitee.username
