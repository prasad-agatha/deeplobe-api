from .base import BaseModelSerializer

from deeplobe_api.db.models import MediaAsset


class MediaAssetSerializer(BaseModelSerializer):
    class Meta:
        model = MediaAsset
        fields = ("id", "name", "asset", "class_name")
