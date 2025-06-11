from .base import BaseModelSerializer

from deeplobe_api.db.models import AnnotationHireExpertRequest


class AnnotationHireExpertRequestSerializer(BaseModelSerializer):
    class Meta:
        model = AnnotationHireExpertRequest
        fields = "__all__"
