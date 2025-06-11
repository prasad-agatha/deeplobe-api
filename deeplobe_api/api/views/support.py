from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404


from deeplobe_api.utils.emails import emails
from deeplobe_api.db.models import Support, SupportTicket, User
from deeplobe_api.api.serializers import (
    SupportSerializer,
    SupportTicketSerializer,
    UserSerializer,
)


class SupportView(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request):

        support = Support.objects.all()
        serializer = SupportSerializer(support, many=True)
        list = []
        for support_data in serializer.data:
            support_ticket = SupportTicket.objects.filter(
                ticket_id=support_data["id"]
            ).first()
            ticket_serializer = SupportTicketSerializer(support_ticket)
            requested_by = User.objects.filter(id=support_data["requested_by"]).first()
            requested_by_serializer = UserSerializer(requested_by)
            data = {}

            data["id"] = support_data["id"]
            data["subject"] = support_data["subject"]
            data["description"] = support_data["description"]
            data["notes"] = support_data["notes"]
            data["file"] = support_data["file"]
            data["status"] = ticket_serializer.data["status"]
            data["name"] = requested_by_serializer.data["username"]
            data["email"] = requested_by_serializer.data["email"]
            data["created"] = support_data["created"]
            data["updated"] = support_data["updated"]
            list.append(data)

        return Response(list, status=status.HTTP_200_OK)

    def post(self, request):
        user = User.objects.get(id=request.user.id)
        name = request.user.username
        email = request.user.email
        subject = request.data.get("subject")
        description = request.data.get("description")
        file = request.data.get("file")
        requested_by = user.id

        dict_data = {
            "name": name,
            "email": email,
            "subject": subject,
            "description": description,
            "file": file,
            "requested_by": requested_by,
        }

        input_subject = request.data.get("subject")
        serializer = SupportSerializer(data=dict_data)
        if serializer.is_valid():
            ticket_id = serializer.save()
            ticket_status = "new"
            SupportTicket.objects.create(ticket_id=ticket_id, status=ticket_status)
            emails.support_email_for_users(request, user, input_subject)
            emails.support_email_for_team(request, input_subject)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SupportDetails(APIView):

    """
    Retrieve, delete a Project
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return Project object if pk value present.
        """
        try:
            return Support.objects.get(pk=pk)
        except Support.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return Project.
        """
        support = self.get_object(pk)
        serializer = SupportSerializer(support)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        support = self.get_object(pk)
        serializer = SupportSerializer(support, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """
        Delete project.
        """
        support = self.get_object(pk)
        support.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


class SupportTicketView(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request):

        support_ticket = SupportTicket.objects.all()

        serializer = SupportSerializer(support_ticket, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def get(self, request, ticket_id):
        support_ticket = SupportTicket.objects.filter(ticket_id=ticket_id)
        serializer = SupportTicketSerializer(support_ticket, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, ticket_id):
        user = User.objects.get(id=request.user.id)
        support_ticket = SupportTicket.objects.get(ticket_id=ticket_id)
        serializer = SupportTicketSerializer(support_ticket, data=request.data)
        if serializer.is_valid():
            serializer.save()
            if support_ticket.status == "resolved":
                emails.support_ticket_closed_email(request, user, ticket_id)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SupportTicketListView(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request):

        support_ticket = SupportTicket.objects.all()

        serializer = SupportTicketSerializer(support_ticket, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)
