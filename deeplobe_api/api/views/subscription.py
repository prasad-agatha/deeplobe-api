import chargebee

from django.http import Http404

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from deeplobe_api.db.models import (
    User,
    Workspace,
    Subscription,
    UserInvitation,
)
from deeplobe_api.utils.emails import emails
from deeplobe_api.api.serializers import UserInvitationSerializer
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails
from deeplobe_api.api.views.function_calls.subscription import deletecollaborators


class SubscriberCollaboratorView(APIView):
    permission_classes = [IsAuthenticated]

    def get_user(self, email):
        return User.objects.filter(email=email).first()

    def get(self, request):
        subscription_collaborator = UserInvitation.objects.filter(
            is_collaborator=False, collaborator_email=request.user.email
        )
        serializer = UserInvitationSerializer(subscription_collaborator, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        role = request.data.get("role")
        models = request.data.get("models")
        workspace = Workspace.objects.filter(user=request.user).first()
        subscription_plan = chargebeeplandetails(email=request.user)
        subscription_collaborator_count = UserInvitation.objects.filter(
            workspace=workspace.id, is_collaborator=True, role="collaborator"
        ).count()
        subscription = request.user.stripe_customer_id
        if subscription_collaborator_count < subscription_plan["users"] - 1 or (
            role == "annotator" and subscription_plan["plan"] != "Free-plan"
        ):
            collaborator_email = request.data.get("email")
            if self.get_user(collaborator_email) is None:
                emails.signup_new_users_email(collaborator_email)
                if not UserInvitation.objects.filter(
                    collaborator_email=collaborator_email,
                    invitee=request.user,
                ).exists():
                    UserInvitation.objects.create(
                        subscription=subscription,
                        workspace=workspace,
                        invitee=request.user,
                        collaborator_email=collaborator_email,
                        role=role,
                        models=models,
                    )

                return Response(
                    "Invitation mail sent successfully", status=status.HTTP_201_CREATED
                )
            else:
                if UserInvitation.objects.filter(
                    collaborator_email=collaborator_email,
                    invitee=request.user,
                    is_collaborator=True,
                ).exists():
                    return Response(
                        "User already exist", status=status.HTTP_400_BAD_REQUEST
                    )
                user_invitation = UserInvitation.objects.filter(
                    collaborator_email=collaborator_email, invitee=request.user
                ).first()

                if not user_invitation:
                    UserInvitation.objects.create(
                        subscription=subscription,
                        workspace=workspace,
                        invitee=request.user,
                        collaborator_email=collaborator_email,
                        role=role,
                        models=models,
                    )
                    emails.user_invitation_collaborator_email(
                        request, subscription, workspace, role
                    )
                    return Response(
                        "Invitation mail sent successfully",
                        status=status.HTTP_201_CREATED,
                    )
                else:
                    user_invitation.role = role
                    user_invitation.models = models
                    user_invitation.save()

                    emails.user_invitation_collaborator_email(
                        request, subscription, workspace, role
                    )
                    return Response(
                        "Invitation mail sent successfully",
                        status=status.HTTP_201_CREATED,
                    )

        else:
            return Response("your plan has exceed ", status=status.HTTP_403_FORBIDDEN)


class SubscriberCollaboratorDetail(APIView):

    """
    Retrieve, delete a Project
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return Project object if pk value present.
        """
        try:
            return UserInvitation.objects.get(id=pk)
        except UserInvitation.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return Project.
        """
        subscriptioncollaborator = self.get_object(pk)
        serializer = UserInvitationSerializer(subscriptioncollaborator)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        subscriptioncollaborator = self.get_object(pk)

        role = request.data.get("role")
        workspace = Workspace.objects.filter(user=request.user).first()
        subscription_plan = chargebeeplandetails(email=request.user)
        subscription_collaborator_count = UserInvitation.objects.filter(
            workspace=workspace.id, is_collaborator=True, role="collaborator"
        ).count()
        if (
            subscription_collaborator_count < subscription_plan["users"] - 1
            or role == "annotator"
        ):
            serializer = UserInvitationSerializer(
                subscriptioncollaborator, data=request.data, partial=True
            )
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        else:
            return Response("your plan has exceed ", status=status.HTTP_403_FORBIDDEN)

    def delete(self, request, pk, format=None):
        """
        Delete project.
        """
        workspace = Workspace.objects.filter(user=request.user).first()
        collaborator_details = User.objects.get(id=pk)
        collaborator = UserInvitation.objects.filter(
            workspace=workspace, collaborator_email=collaborator_details.email
        ).first()
        subscriptioncollaborator = self.get_object(collaborator.id)
        subscriptioncollaborator.delete()
        user = User.objects.filter(id=pk).first()
        if workspace.id == user.current_workspace:
            collaborator_workspace = Workspace.objects.filter(user=user).first()
            user.current_workspace = collaborator_workspace.id
            user.save()

        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


class CollaboratorInvitation(APIView):
    permission_classes = [IsAuthenticated]

    def get_user(self, email):
        return User.objects.filter(email=email).first()

    def post(self, request):
        token = request.data.get("token")
        workspace_id = request.data.get("workspace")
        email = request.user
        workspace = Workspace.objects.filter(id=workspace_id).first()
        subscription_plan = chargebeeplandetails(email=workspace.user)
        subscription_collaborator_count = UserInvitation.objects.filter(
            workspace=workspace.id, is_collaborator=True, role="collaborator"
        ).count()
        collaborator = UserInvitation.objects.filter(
            subscription=token, collaborator_email=email
        ).first()
        if collaborator is None:
            return Response(
                "please provide valid token or email id",
                status=status.HTTP_400_BAD_REQUEST,
            )

        if (
            subscription_collaborator_count < subscription_plan["users"] - 1
            or collaborator.role == "annotator"
        ):
            collaborator.is_collaborator = True
            collaborator.save()
            UserInvitation.objects.filter(
                collaborator_email=email, invitee=collaborator.invitee
            )
            # add user to workspace
            return Response("invitation accepted", status=status.HTTP_200_OK)
        else:
            return Response(
                "collaborators exceeded for this subscription",
                status=status.HTTP_403_FORBIDDEN,
            )

    def delete(self, request):
        token = request.query_params.get("token")
        email = request.user
        collaborator_name = request.user.username
        collaborator = UserInvitation.objects.filter(
            subscription=token, collaborator_email=email
        ).first()
        if collaborator:
            user_invitation = UserInvitation.objects.filter(
                collaborator_email=email, invitee=collaborator.invitee
            )
            user_invitation.delete()
            emails.collaborator_invitation_rejected_email(
                collaborator, collaborator_name
            )
            return Response("Invitation rejected", status=status.HTTP_200_OK)
        else:
            return Response(
                "please provide valid token or email id",
                status=status.HTTP_400_BAD_REQUEST,
            )


class WorkspaceCollaboratorsDeletion(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request):
        if request.user.stripe_subcription:
            deletecollaborators(request)
            return Response(
                "Collaborators deleted successfully", status=status.HTTP_200_OK
            )

        return Response(
            "Error deleting collaborators", status=status.HTTP_400_BAD_REQUEST
        )
