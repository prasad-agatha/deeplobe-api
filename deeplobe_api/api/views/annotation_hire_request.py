from django.http import Http404

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from deeplobe_api.utils.emails import emails
from deeplobe_api.db.models import AnnotationHireExpertRequest, User
from deeplobe_api.api.serializers import (
    AnnotationHireExpertRequestSerializer,
    UserSerializer,
)


class AnnotationHireExpertRequestView(APIView):

    """
    API endpoint that allows view list of all the annotation hire expert request or create new annotation hire expert request.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of all annotation hire expert request.
        """

        annotation_hire_expert_request = AnnotationHireExpertRequest.objects.all()
        serializer = AnnotationHireExpertRequestSerializer(
            annotation_hire_expert_request, many=True
        )
        list = []
        for annotation_hire_data in serializer.data:
            requested_by = User.objects.filter(
                id=annotation_hire_data["requested_by"]
            ).first()
            requested_by_serializer = UserSerializer(requested_by)
            data = {}
            data["id"] = annotation_hire_data["id"]
            data["subject"] = annotation_hire_data["subject"]
            data["description"] = annotation_hire_data["description"]
            data["notes"] = annotation_hire_data["notes"]
            data["status"] = annotation_hire_data["status"]
            data["name"] = requested_by_serializer.data["username"]
            data["email"] = requested_by_serializer.data["email"]
            data["created"] = annotation_hire_data["created"]
            data["updated"] = annotation_hire_data["updated"]
            data["status"] = annotation_hire_data["status"]
            list.append(data)

        return Response(list, status=status.HTTP_200_OK)

    def post(self, request):
        """
        Create a new annotation hire expert request.
        """
        user = User.objects.get(id=request.user.id)
        name = request.user.username
        email = request.user.email
        subject = request.data.get("subject")
        description = request.data.get("description")
        requested_by = user.id

        dict_data = {
            "name": name,
            "email": email,
            "subject": subject,
            "description": description,
            "requested_by": requested_by,
        }

        serializer = AnnotationHireExpertRequestSerializer(data=dict_data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            emails.hire_annotation_expert_email(request, user)
            return Response(serializer.data)


class AnnotationHireExpertRequestDetails(APIView):

    """
    API endpoint that allows view, update, delete individual annotation hire expert request details.

    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return annotation hire expert request object if pk value present.
        """
        try:
            return AnnotationHireExpertRequest.objects.get(pk=pk)
        except AnnotationHireExpertRequest.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return annotation hire expert request.
        """
        annotation_hire_expert_request = self.get_object(pk)
        serializer = AnnotationHireExpertRequestSerializer(
            annotation_hire_expert_request
        )
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        """
        Update annotation hire expert request object if pk value present.
        """
        annotation_hire_expert_request = self.get_object(pk)
        serializer = AnnotationHireExpertRequestSerializer(
            annotation_hire_expert_request, data=request.data
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """
        Delete annotation hire expert request object if pk value present.
        """
        annotation_hire_expert_request = self.get_object(pk)
        annotation_hire_expert_request.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
