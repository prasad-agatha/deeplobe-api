from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404, FileResponse

from deeplobe_api.utils.emails import emails
from deeplobe_api.db.models import Task, FileAssets


class MyModels(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request):

        data = Task.objects.filter(
            user=request.user, task_type="training", task_finished=True
        )

        list_data = []

        for i in data:
            weight_file = FileAssets.objects.filter(
                uuid=i.uuid, photo__endswith=".ckpt"
            )
            if weight_file:
                file_url = (
                    request.scheme
                    + "://"
                    + request.META["HTTP_HOST"]
                    + weight_file[0].photo.url
                )
            else:
                file_url = None
            list_data.append(
                {
                    "name": i.name,
                    "description": i.description,
                    "weight_name": i.weight_name,
                    "uuid": i.uuid,
                    "file_url": file_url,
                    "created": i.created,
                    "updated": i.updated,
                }
            )
        return Response(list_data, status=status.HTTP_200_OK)


class ModelDetail(APIView):
    """
    Retrieve, delete a user
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, uuid):
        """
        Return user object if pk value present.
        """
        try:
            return Task.objects.get(uuid=uuid)
        except Task.DoesNotExist:
            raise Http404

    def delete(self, request, uuid, format=None):
        """
        Delete user.
        """
        try:
            data = Task.objects.filter(
                user=request.user, task_type="training", task_finished=True
            )

            for i in data:
                weight_file = FileAssets.objects.filter(
                    uuid=i.uuid, photo__endswith=".ckpt"
                ).first()
                if weight_file:
                    file_url = (
                        request.scheme
                        + "://"
                        + request.META["HTTP_HOST"]
                        + weight_file.photo.url
                    )

            emails.model_deactivate_or_delete_email(request, weight_file)
            model = self.get_object(uuid)

            model.delete()
            return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ModelDownload(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request, uuid):
        weight_file = FileAssets.objects.filter(uuid=uuid, photo__endswith=".ckpt")
        if weight_file:
            response = FileResponse(open(weight_file[0].photo.path, "rb"))
            return response
        return Response(
            {"error": "File not exists in db."}, status=status.HTTP_400_BAD_REQUEST
        )
