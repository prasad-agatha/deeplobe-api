from .base import BaseModelSerializer

from deeplobe_api.db.models import ContactUs


class ContactUsSerializer(BaseModelSerializer):
    class Meta:
        model = ContactUs
        fields = [
            "id",
            "contact_number",
            "name",
            "description",
            "hearing",
            "email",
            "role",
            "company",
            "created",
            "updated",
            "title",
            "status",
            "notes",
            "requested_by",
        ]
