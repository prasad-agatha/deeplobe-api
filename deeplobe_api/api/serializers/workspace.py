from .base import BaseModelSerializer

from rest_framework import serializers

from deeplobe_api.db.models import Workspace


class WorkspaceSerializer(BaseModelSerializer):
    user = serializers.StringRelatedField()

    class Meta:
        model = Workspace
        fields = "__all__"
