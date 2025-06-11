from .base import BaseModelSerializer

from deeplobe_api.db.models import Register


class RegisterSerializer(BaseModelSerializer):
    class Meta:
        model = Register
        fields = "__all__"
