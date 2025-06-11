from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404

from deeplobe_api.utils.emails import emails
from deeplobe_api.db.models import ContactUs, User
from deeplobe_api.api.serializers import ContactUsSerializer, UserSerializer


class ContactUsViewSet(APIView):

    """
    API endpoint that allows user to create contact request or view the list of all contact request.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):

        """
        Return a list of all contact request.
        """

        contact_us = ContactUs.objects.all()

        serializer = ContactUsSerializer(contact_us, many=True)
        list = []
        for contact_data in serializer.data:
            requested_by = User.objects.filter(id=contact_data["requested_by"]).first()
            requested_by_serializer = UserSerializer(requested_by)
            data = {}

            data["id"] = contact_data["id"]
            data["subject"] = contact_data["hearing"]
            data["description"] = contact_data["description"]
            data["notes"] = contact_data["notes"]
            data["status"] = contact_data["status"]
            data["name"] = requested_by_serializer.data["username"]
            data["email"] = requested_by_serializer.data["email"]
            data["created"] = contact_data["created"]
            data["updated"] = contact_data["updated"]
            list.append(data)

        return Response(list, status=status.HTTP_200_OK)

    def post(self, request):

        """
        Create a new contact request.
        """

        user = User.objects.get(id=request.user.id)
        name = request.user.username
        email = request.user.email
        description = request.data.get("description")
        contact_number = request.data.get("contact_number")
        hearing = request.data.get("hearing")
        title = request.data.get("title")
        requested_by = user.id

        dict_data = {
            "name": name,
            "email": email,
            "description": description,
            "requested_by": requested_by,
            "contact_number": contact_number,
            "hearing": hearing,
            "title": title,
        }
        serializer = ContactUsSerializer(data=dict_data)
        if serializer.is_valid():
            serializer_data = serializer.save()
            emails.contact_us_email_for_users(request, serializer_data, email)
            if serializer_data.title == "API implementation request":
                emails.api_implementation_request_contact_us_team_email(
                    request, serializer_data
                )
            else:
                emails.custom_model_request_contact_us_team_email(
                    request, serializer_data
                )
            return Response(
                serializer.data,
                # {"info": "Your query submitted, management will reach you!"},
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ContactUsDetails(APIView):

    """
    API endpoint that allows view, update, delete individual contact request details.

    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return contact request object if pk value present.
        """
        try:
            return ContactUs.objects.get(pk=pk)
        except ContactUs.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return contact request.
        """
        contact_us = self.get_object(pk)
        serializer = ContactUsSerializer(contact_us)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):

        """
        Update contact request object if pk value present.
        """
        contact_us = self.get_object(pk)
        serializer = ContactUsSerializer(contact_us, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):

        """
        Delete contact request object if pk value present.
        """
        contact_us = self.get_object(pk)
        contact_us.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
