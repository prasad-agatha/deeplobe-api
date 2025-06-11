from .base import BaseModelSerializer

from deeplobe_api.db.models import AutoAnnotatePredictionModel


class AutoAnnotatePredictionModelSerializer(BaseModelSerializer):
    class Meta:
        model = AutoAnnotatePredictionModel
        fields = "__all__"
