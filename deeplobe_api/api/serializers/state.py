from .base import BaseModelSerializer

from deeplobe_api.db.models import State


class StateSerializer(BaseModelSerializer):
    class Meta:
        model = State
        fields = ["name", "parent_name"]
