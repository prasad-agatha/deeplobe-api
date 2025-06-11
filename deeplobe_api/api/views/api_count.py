from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404
from django.db.models import Count, Max

from deeplobe_api.db.models import (
    APILogger,
    APICount,
    Workspace,
    AIModel,
    UserInvitation,
)
from deeplobe_api.api.serializers import APICountSerializer
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails


class APICountList(APIView):

    """
    API endpoint that allows to view list of count in current workspace

    * Requires JWT authentication.
    * This endpoint will allows only GET methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of all count.
        """
        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()
        qs = (
            APILogger.objects.filter(workspace=workspace)
            .values("model_type")
            .annotate(model_type_count=Count("model_type"), created=Max("created"))
        )
        pretrained_count = qs.count()
        custom_model_count = AIModel.objects.filter(
            workspace=request.user.current_workspace
        ).count()
        metadata = chargebeeplandetails(email=workspace.user)
        collaborators = UserInvitation.objects.filter(
            workspace=workspace, is_collaborator=True, role="collaborator"
        ).count()

        apicount = APICount.objects.filter(
            workspace=workspace.id,
        )
        if apicount.exists():
            apicalls = (
                apicount[0].custom_model_api_count
                + APILogger.objects.filter(api_key__user=request.user).count()
            )

        else:
            apicalls = APILogger.objects.filter(api_key__user=request.user).count()

        metadata["api_calls"] = apicalls
        metadata["collaborators"] = collaborators
        if metadata["plan"] == "DeepLobe-Free-Plan-USD-Monthly":
            metadata["active_models"] = pretrained_count
        else:
            metadata["active_models"] = pretrained_count + custom_model_count

        return Response(metadata, status=status.HTTP_200_OK)


class APICountDetail(APIView):

    """
    API endpoint that allows view, update, delete individual count.

    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return count object if pk value present.
        """

        try:
            return APICount.objects.get(pk=pk)
        except APICount.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return count object.
        """

        apicount = self.get_object(pk)
        serializer = APICountSerializer(apicount)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk, format=None):
        """
        Delete count object if pk value present.
        """

        apicount = self.get_object(pk)
        apicount.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
