from .base import BaseModelSerializer

from deeplobe_api.db.models.support import Support, SupportTicket


class SupportSerializer(BaseModelSerializer):
    class Meta:
        model = Support
        fields = "__all__"


class SupportTicketSerializer(BaseModelSerializer):
    class Meta:
        model = SupportTicket
        fields = "__all__"
