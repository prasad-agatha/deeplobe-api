from .base import BaseModelSerializer

from deeplobe_api.db.models import APICount


class APICountSerializer(BaseModelSerializer):
    class Meta:
        model = APICount
        fields = "__all__"
