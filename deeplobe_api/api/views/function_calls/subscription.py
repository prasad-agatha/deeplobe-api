from rest_framework import status
from rest_framework.response import Response

from deeplobe_api.db.models import (
    User,
    Workspace,
    Subscription,
    UserInvitation,
)
from deeplobe_api.api.serializers import UserSerializer


from deeplobe_api.utils.emails import emails
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails


def new_user_invitations(user, email):
    user_invitations = UserInvitation.objects.filter(
        collaborator_email=email, is_collaborator=False
    )

    for new_user_invitation in user_invitations:
        invitee_user = User.objects.filter(email=new_user_invitation.invitee).first()
        serializer = UserSerializer(invitee_user)
        invitee_user_data = serializer.data
        # collaborator_user = User.objects.filter(
        #     email=new_user_invitation.collaborator_email
        # ).first()
        invitee_subscription = invitee_user_data.stripe_customer_id
        invitee_workspace = Workspace.objects.filter(user=invitee_user).first()
        invitee_subscription_plan = chargebeeplandetails(email=invitee_user)
        invitee_subscription_collaborator_count = UserInvitation.objects.filter(
            workspace=invitee_workspace.id, is_collaborator=True, role="collaborator"
        ).count()
        if (
            invitee_subscription_collaborator_count
            < invitee_subscription_plan["users"] - 1
            or new_user_invitation.role == "annotator"
        ):
            emails.new_user_invitation_collaborator_email(
                user,
                invitee_subscription,
                invitee_workspace,
                new_user_invitation,
            )


def deletecollaborators(self, request):
    try:
        workspace = Workspace.objects.filter(user=request.user).first()
        subscription_collaborators = UserInvitation.objects.filter(
            workspace=workspace, is_collaborator=True
        )
        for collaborator in subscription_collaborators:
            if collaborator.collaborator.current_workspace == workspace.id:
                user = User.objects.filter(id=collaborator.collaborator.id).first()
                collaborator_workspace = Workspace.objects.filter(user=user).first()
                user.current_workspace = collaborator_workspace.id
                user.save()

            collaborator.delete()
        return Response("Deleted succesfully", status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
