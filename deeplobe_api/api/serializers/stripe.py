from .base import BaseModelSerializer

from deeplobe_api.db.models import StripeSubcription, SubcriptionInvoices


class StripeSerializer(BaseModelSerializer):
    class Meta:
        model = StripeSubcription
        fields = "__all__"


class SubscriptionInvoiceSerializer(BaseModelSerializer):
    class Meta:
        model = SubcriptionInvoices
        fields = "__all__"
