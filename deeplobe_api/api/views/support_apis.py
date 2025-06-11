from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from deeplobe_api.db.models import FileAssets, AIModel


class UuidStatus(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request, uuid):

        data = FileAssets.objects.filter(uuid=uuid).values("category_name").distinct()

        return Response({"data": data}, status=status.HTTP_200_OK)


class AllCategories(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request):

        data = (
            FileAssets.objects.filter(model_name="classification")
            .values("category_name")
            .distinct()
        )

        return Response({"data": data}, status=status.HTTP_200_OK)


class CheckName(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request, name):

        data = AIModel.objects.filter(name=name, user=request.user)
        if data:
            return Response(
                {"msg": "Name present", "status": "True"}, status=status.HTTP_200_OK
            )
        else:
            return Response(
                {"msg": "Name Not present", "status": "False"},
                status=status.HTTP_200_OK,
            )
