from .base import BaseModelSerializer

from deeplobe_api.db.models import FileAssets


class FileUploadSerializer(BaseModelSerializer):
    class Meta:
        model = FileAssets
        fields = [
            "id",
            "model_name",
            "uuid",
            "photo",
            "category_name",
            "user",
            "created",
            "updated",
        ]
