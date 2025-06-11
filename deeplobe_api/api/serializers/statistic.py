from .base import BaseModelSerializer

from deeplobe_api.db.models import Statistic


class StatisticSerializer(BaseModelSerializer):
    class Meta:
        model = Statistic
        fields = "__all__"
