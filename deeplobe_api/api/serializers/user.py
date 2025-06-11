from .base import BaseModelSerializer

from deeplobe_api.db.models import User


class UserSerializer(BaseModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "company",
            "job_title",
            "terms_conditions",
            "role",
            "industry",
            "contact_number",
            "profile_pic",
            "last_login",
            "is_new",
            "is_active",
            "is_admin",
            "interestedIn",
            "model_permissions",
            "created",
            "updated",
            "extended_date",
            "current_workspace",
            "postal_code",
            "country",
            "state",
            "GSTIN",
            "company_size",
            "address",
            "bussiness_email",
            "help",
            "notes",
            "stripe_subcription",
            "stripe_customer_id",
            "stripe_currency",
        ]


class UserProfilePicSerializer(BaseModelSerializer):
    class Meta:
        model = User
        fields = ["username", "profile_pic"]
