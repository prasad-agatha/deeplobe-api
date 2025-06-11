from django.http import Http404
from django.conf import settings

from decouple import config

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated


from deeplobe_api.db.models import APIKey,Workspace
from deeplobe_api.api.serializers import APIKeySerializer
from deeplobe_api.api.views.function_calls.api_key import create_api_key
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails


class APIKeyEndpoint(APIView):

    """
    API endpoint that allows user to create apikeys or view the list of all apikeys.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of all apikeys.
        """
        user = request.user
        q = request.query_params.get("application_name")
        if q is not None:
            api_keys = user.api_keys.filter(application_name__icontains=q)
        else:
            api_keys = user.api_keys.all()
        serializer = APIKeySerializer(api_keys, many=True)
        return Response(serializer.data)

    def post(self, request):
        """
        Create a new apikey.
        """
        
        workspace_id = request.user.current_workspace
        workspace_obj = Workspace.objects.filter(id=workspace_id).first()
        if (
            chargebeeplandetails(email=workspace_obj.user).get("plan")
            == config("STRIPE_FREE_PLAN")
        ):
            return create_api_key(request, 1)
        return create_api_key(request, 6)


class UserAPIKeysDetail(APIView):

    """
    API endpoint that allows view, update, delete individual apikey details.

    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated,)

    def get_object(self, pk):
        """
        Return apikey object if pk value present.
        """
        try:
            return APIKey.objects.get(pk=pk)
        except APIKey.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        """
        Return apikey.
        """

        api_key = self.get_object(pk)
        serializer = APIKeySerializer(api_key)
        return Response(serializer.data)

    def put(self, request, pk):
        """
        Update apikey object if pk value present.
        """
        api_key = self.get_object(pk)
        serializer = APIKeySerializer(api_key, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        Delete apikey object if pk value present.
        """

        api_key = self.get_object(pk)
        api_key.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
