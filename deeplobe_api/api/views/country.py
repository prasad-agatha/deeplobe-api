from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404

from deeplobe_api.db.models import Country
from deeplobe_api.api.views.filters import CountryFilter
from deeplobe_api.api.serializers import CountrySerializer


class CountryList(APIView):

    """
    API endpoint that allows user to create country or view the list of all country.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):

        """
        Return a list of all country.
        """
        country = CountryFilter(request.GET, queryset=Country.objects.all()).qs
        serializer = CountrySerializer(country, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):

        """
        Create a new country.
        """
        serializer = CountrySerializer(data=request.data, many=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)


class CountryDetail(APIView):

    """
    API endpoint that allows view, update, delete individual country details.

    * Requires JWT authentication.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):

        """
        Return country object if pk value present.
        """
        try:
            return Country.objects.get(pk=pk)
        except Country.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        """
        Return country.
        """
        country = self.get_object(pk)
        serializer = CountrySerializer(country)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk, format=None):

        """
        Delete country object if pk value present.
        """
        country = self.get_object(pk)
        country.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
