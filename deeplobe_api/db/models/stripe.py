from django.db import models
from django.db.models.fields import json
from ..mixins import TimeAuditModel
from django.db.models import JSONField
from django.conf import settings


class StripeSubcription(TimeAuditModel):
    stripe_customer_id = models.CharField(max_length=255)
    stripe_subscription_id = models.CharField(max_length=255, unique=True)
    customer_email = models.CharField(max_length=255, null=True, blank=True)
    stripe_subscription = models.JSONField(default=dict, null=True, blank=True)
    metadata = models.JSONField(max_length=255, null=True, blank=True)
    status = models.CharField(max_length=255, null=True, blank=True)
    plan_name = models.CharField(max_length=255, null=True, blank=True)
    price = models.JSONField(max_length=255, null=True, blank=True)
    quantity = models.IntegerField(null=True, blank=True)
    canceled_at = models.CharField(max_length=255, null=True, blank=True)
    current_period_start = models.CharField(max_length=255, null=True, blank=True)
    current_period_end = models.CharField(max_length=255, null=True, blank=True)
    created = models.CharField(max_length=255, null=True, blank=True)
    ended_at = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        verbose_name = "StripeSubcription"
        verbose_name_plural = " StripeSubcriptions"
        db_table = "stripesubcription"

    def __str__(self):
        return f"{self.customer_email} <{self.plan_name}>"


class SubcriptionInvoices(TimeAuditModel):
    subscription = models.ForeignKey(StripeSubcription, on_delete=models.CASCADE)
    invoice_details = models.JSONField(max_length=255, null=True, blank=True)

    class Meta:
        verbose_name = "SubscriptionInvoices"
        verbose_name_plural = " SubscriptionInvoices"
        db_table = "subscriptionsinvoice"

    def __str__(self):
        return f"{self.subscription.customer_email} <{self.subscription.plan_name}| {self.created}>"
