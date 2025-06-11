from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404

from deeplobe_api.db.models import State
from deeplobe_api.api.views.filters import StateFilter
from deeplobe_api.api.serializers import StateSerializer


class StateList(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request):
        country = (StateFilter(request.GET, queryset=State.objects.all()).qs).filter(
            parent_name=request.query_params.get("country")
        )
        serializer = StateSerializer(country, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = StateSerializer(data=request.data, many=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class StateDetail(APIView):

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return Project object if pk value present.
        """
        try:
            return State.objects.get(pk=pk)
        except State.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return Project.
        """
        state = self.get_object(pk)
        serializer = StateSerializer(state)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk, format=None):
        """
        Delete project.
        """
        state = self.get_object(pk)
        state.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
