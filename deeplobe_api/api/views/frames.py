import json
import requests

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.base import ContentFile

from deeplobe_api.db.models import AIModel, MediaAsset
from deeplobe_api.api.views.function_calls.image import raw_resize_image


from deeplobe_api.api.views.function_calls.image import (
    frames_count,
    # get_frames,
    extract_frames,
    frames_to_video,
    delete_object_in_s3,
    save_django_file_in_s3,
)


class FramesCountAPIView(APIView):
    def post(self, request):
        video_file = request.FILES.get("video")
        # gfps = request.data.get("gfps", None)
        url, key = save_django_file_in_s3(video_file)
        response_data = frames_count(url)
        return Response(response_data)


class FramesAPIView(APIView):
    def post(self, request):
        video_file = request.FILES.get("video")
        gfps = request.data.get("gfps", None)
        # convert binary video to s3 video
        url, key = save_django_file_in_s3(video_file)

        if float(gfps):
            width = int(request.data.get("width", None))
            height = int(request.data.get("height", None))
            uuid = request.data.get("uuid", None)
            frames = extract_frames(url, gfps, height, width, uuid)
            urls = frames
            # get Aimodel
            aimodel = AIModel.objects.filter(uuid=uuid).first()
            if aimodel is not None:
                annotation_file = aimodel.annotation_file
                if annotation_file:
                    response = requests.get(annotation_file.url).json()
                else:
                    response = {"categories": [], "annotations": [], "images": []}
                images = response["images"]
                for url in urls:
                    images.append(
                        {
                            "id": url["id"],
                            "file_name": url["name"],
                            "url": url["asset"],
                            "annotated": False,
                        }
                    )
                aimodel.annotation_file.save(
                    "Annotate.json",
                    ContentFile(json.dumps(response, indent=2).encode("utf-8")),
                    save=True,
                )
                aimodel.save()
            delete_object_in_s3(key)
            # frames = get_frames(video_file, images_count)
            return Response({"msg": "Successfully uploaded"})
        return Response({"msg": "Bad Request"}, status=status.HTTP_400_BAD_REQUEST)


class FramesToVideoAPIView(APIView):
    def post(self, request):
        url = frames_to_video(request.data.get("urls"))
        return Response({"msg": url})


class UpdateAnnotionsWithS3URLsAPIView(APIView):
    def post(self, request):
        import uuid
        import io

        input_uuid = request.data.get("uuid", None)
        width = request.data.get("width", None)
        height = request.data.get("height", None)

        urls = request.data.get("urls", None)
        aimodel = AIModel.objects.filter(uuid=input_uuid).first()
        assets = []
        for url in urls:
            name = f"{uuid.uuid4().hex}.jpg"
            # img = requests.get(url=url,stream=True).content
            print(url, "URL")
            response = requests.get(url, stream=True)
            buffer = io.BytesIO(response.content)
            resized_image = raw_resize_image(
                img=buffer,
                input_width=int(width),
                input_height=int(height),
                buffer=True,
            )
            # content_file = ContentFile(requests.get(url=url).content, name=name)
            asset = MediaAsset.objects.create(name=name, asset=resized_image)
            assets.append(asset)
        if aimodel is not None:
            annotation_file = aimodel.annotation_file
            if annotation_file:
                response = requests.get(annotation_file.url).json()
            else:
                response = {"categories": [], "annotations": [], "images": []}
            images = response["images"]
            for asset in assets:
                images.append(
                    {
                        "id": asset.id,
                        "file_name": asset.name,
                        "url": asset.asset.url,
                        "annotated": False,
                    }
                )
            print(response, "RESPONCE")
            aimodel.annotation_file.save(
                "Annotate.json",
                ContentFile(json.dumps(response, indent=2).encode("utf-8")),
                save=True,
            )
            aimodel.save()
        # frames = get_frames(video_file, images_count)
        return Response({"msg": "Successfully uploaded"})
