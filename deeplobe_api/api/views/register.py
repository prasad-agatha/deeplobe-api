from django.http import Http404

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from deeplobe_api.db.models import Register
from deeplobe_api.api.serializers import RegisterSerializer


class RegisterList(APIView):
    """
    A view for viewing and Employee.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of all Register.
        """
        queryset = Register.objects.all()
        serializer = RegisterSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        """
        Create a register.
        """

        serializer = RegisterSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RegisterDetail(APIView):
    """
    Retrieve, delete a Register
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return blog object if pk value present.
        """
        try:
            return Register.objects.get(pk=pk)
        except Register.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return Register.
        """
        register = self.get_object(pk)

        serializer = RegisterSerializer(register)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk, format=None):

        register = self.get_object(pk)
        serializer = RegisterSerializer(register, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """
        Delete register.
        """
        register = self.get_object(pk)
        register.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
