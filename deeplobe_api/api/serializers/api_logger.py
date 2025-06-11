from .base import BaseModelSerializer

from rest_framework import serializers

from deeplobe_api.db.models import APILogger


class APILoggerSerializer(BaseModelSerializer):
    class Meta:
        model = APILogger
        fields = "__all__"
