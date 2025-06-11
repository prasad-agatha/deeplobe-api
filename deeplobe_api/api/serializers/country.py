from .base import BaseModelSerializer

from deeplobe_api.db.models.country import Country


class CountrySerializer(BaseModelSerializer):
    class Meta:
        model = Country
        fields = ["name"]
