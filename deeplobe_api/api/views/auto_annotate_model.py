from django.http import Http404

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from deeplobe_ai.auto_annotate import get_prediction

from deeplobe_api.db.models import AutoAnnotatePredictionModel
from deeplobe_api.api.serializers import AutoAnnotatePredictionModelSerializer


class AutoAnnotateListView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        dict_data = {
            "input_images": request.data.get("images"),
            "user": request.user.id,
        }
        input_images = get_prediction(request.data.get("images"))
        dict_data["result"] = input_images
        serializer = AutoAnnotatePredictionModelSerializer(data=dict_data)
        if serializer.is_valid():
            serializer.save()

            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AutoAnnotateDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        try:
            return AutoAnnotatePredictionModel.objects.get(pk=pk)
        except AutoAnnotatePredictionModel.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        auto_annotate = self.get_object(pk)
        serializer = AutoAnnotatePredictionModelSerializer(auto_annotate)
        return Response(serializer.data)

    def put(self, request, pk):
        auto_annotate = self.get_object(pk)
        serializer = AutoAnnotatePredictionModelSerializer(
            auto_annotate, data=request.data
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        auto_annotate = self.get_object(pk)
        auto_annotate.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
