from rest_framework import serializers

from deeplobe_api.db.models import Subscription, UserInvitation


class SubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscription
        fields = "__all__"


class UserInvitationSerializer(serializers.ModelSerializer):
    subscription = serializers.SerializerMethodField(read_only=True)
    invitee = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = UserInvitation
        fields = "__all__"

    def get_invitee(self, obj):
        return {"email": obj.invitee.email, "username": obj.invitee.username}

    def get_subscription(self, obj):
        return obj.subscription
