from .base import BaseModelSerializer

from rest_framework import serializers

from deeplobe_api.db.models import APIKey


class APIKeySerializer(BaseModelSerializer):
    # expire_date = serializers.DateField(read_only=True)

    class Meta:
        model = APIKey
        fields = "__all__"
