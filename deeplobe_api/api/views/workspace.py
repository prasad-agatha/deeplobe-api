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
from deeplobe_api.api.serializers import (
    WorkspaceSerializer,
    UserSerializer,
    UserInvitationSerializer,
)
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails
from deeplobe_api.api.views.function_calls.subscription import deletecollaborators


class WorkspaceView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        personal_workspace = Workspace.objects.filter(user=request.user).first()

        serializer = WorkspaceSerializer(personal_workspace)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        request.data["user"] = request.user.id
        serializer = WorkspaceSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class WorkspaceDetail(APIView):

    """
    Retrieve, delete a Project
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return Project object if pk value present.
        """
        try:
            return Workspace.objects.get(pk=pk)
        except Workspace.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return Project.
        """
        workspace = self.get_object(pk)
        serializer = WorkspaceSerializer(workspace)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk, format=None):
        workspace = self.get_object(pk)
        serializer = WorkspaceSerializer(workspace, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """
        Delete project.
        """
        workspace = self.get_object(pk)
        workspace.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


class WorkspaceUsersView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        personal_workspace = Workspace.objects.filter(user=request.user).first()

        if not request.user.stripe_subcription:
            deletecollaborators(self, request)
        subscription_collaborators = UserInvitation.objects.filter(
            workspace=personal_workspace, is_collaborator=True
        )

        collaborators = []
        annotators = []

        for collaborator in subscription_collaborators:
            collaborator_data = UserInvitationSerializer(collaborator).data
            inv_user = User.objects.filter(
                email=collaborator.collaborator_email
            ).first()
            inv_user_data = UserSerializer(inv_user).data
            result = {
                "id": inv_user_data["id"],
                "collaborator_id": collaborator_data["id"],
                "name": inv_user_data["username"],
                "email": inv_user_data["email"],
                "is_active": inv_user_data["is_active"],
                "role": collaborator.role,
                "models": collaborator.models,
            }

            if collaborator.role == "collaborator":
                collaborators.append(result)
            if collaborator.role == "annotator":
                annotators.append(result)

        response = {}
        owner = UserSerializer(request.user).data
        response["owner"] = {
            "id": owner["id"],
            "name": owner["username"],
            "email": owner["email"],
            "is_active": owner["is_active"],
            "role": "owner",
            "models": [
                {
                    "select": ["All"],
                    "api": True,
                    "create_model": True,
                    "delete_model": True,
                    "test_model": True,
                }
            ],
        }
        response["collaborators"] = collaborators
        response["annotators"] = annotators

        return Response(response)


class PersonalWorkspaceView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        result = chargebeeplandetails(email=request.user)

        return Response(result, status=status.HTTP_200_OK)
