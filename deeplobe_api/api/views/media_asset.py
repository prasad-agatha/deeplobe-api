from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.shortcuts import get_object_or_404


from deeplobe_api.db.models import MediaAsset
from deeplobe_api.api.views.function_calls.image import resize_image
from deeplobe_api.api.serializers import MediaAssetSerializer


class MediaAssetView(APIView):

    """
    API endpoint that allows upload files into to default file storage(like s3 bucket etc) or view the uploaded files.

    * Authentication not required.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):

        """
        Return a list of all the media files.
        """

        queryset = MediaAsset.objects.all()

        serializer = MediaAssetSerializer(queryset, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):

        """
        Upload file into default file storage.
        """

        myDict = dict(request.data)
        try:
            if len(myDict["name[]"]) != len(myDict["asset[]"]):
                return Response(
                    {"Error": "length of name[], lenght of asset[] not equal"}
                )
            else:
                dict_data = [
                    {"name": name, "asset": asset}
                    for name, asset in zip(myDict["name[]"], myDict["asset[]"])
                ]

        except Exception as e:
            return Response(
                {"Error": "Payload error" + str(e)}, status=status.HTTP_400_BAD_REQUEST
            )
        serializer = MediaAssetSerializer(data=dict_data, many=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MediaAssetDetailView(APIView):

    """
    API endpoint that allows view individual file details or delete individual file.

    * Authentication not required.
    * This endpoint will allows only GET, DELETE methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, pk):

        """
        View individual file details
        """

        queryset = get_object_or_404(MediaAsset, id=pk)
        serializer = MediaAssetSerializer(queryset)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):

        """
        Update individual file details
        """
        queryset = get_object_or_404(MediaAsset, id=pk)
        myDict = dict(request.data)
        dict_data = [
            {
                "name": name,
                "asset": asset,
                "class_name": class_name,
            }
            for name, asset, class_name in zip(
                myDict.get("name"), myDict.get("asset"), myDict.get("class_name")
            )
        ]

        serializer = MediaAssetSerializer(queryset, data=dict_data[0], partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # delete store by id
    def delete(self, request, pk):

        """
        Delete individual file details
        """

        queryset = get_object_or_404(MediaAsset, id=pk)
        queryset.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


class ImageResizeEndpoint(APIView):

    """
    API endpoint that allows update the size of an particular image.

    * Requires JWT authentication.
    * This endpoint will allows only PUT methods.
    """

    permission_classes = [IsAuthenticated]

    def put(self, request, pk):

        """
        Update image size object if pk value present.
        """

        queryset = get_object_or_404(MediaAsset, id=pk)
        serializer = MediaAssetSerializer(queryset)
        try:
            # new image
            # user format will be (width, height)
            resized_image = resize_image(
                url=serializer.data.get("asset", None),
                width=request.data.get("width", None),
                height=request.data.get("height", None),
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        dict_data = {
            "name": queryset.name,
            "asset": resized_image,
            "class_name": queryset.class_name,
        }
        serializer = MediaAssetSerializer(queryset, data=dict_data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)


class ImageResize(APIView):

    permission_classes = [IsAuthenticated]

    def post(self, request, uuid):
        class_name = request.data.get("class_name")
        asset = request.data.get("asset")
        dict_data = {"name": uuid, "asset": asset, "class_name": class_name}

        serializer = MediaAssetSerializer(data=dict_data)
        if serializer.is_valid():
            serializer.save()

            queryset = get_object_or_404(MediaAsset, id=serializer.data.get("id", None))
            media_serializer = MediaAssetSerializer(queryset)
            try:
                # new image
                # user format will be (width, height)
                resized_image = resize_image(
                    url=media_serializer.data.get("asset", None),
                    width=int(request.data.get("width", None)),
                    height=int(request.data.get("height", None)),
                )
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

            dict_data = {
                "name": queryset.name,
                "asset": resized_image,
                "class_name": queryset.class_name,
            }
            serializer = MediaAssetSerializer(queryset, data=dict_data, partial=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
