from django.urls import path

from deeplobe_api.api.views.frames import (
    FramesAPIView,
    FramesCountAPIView,
    FramesToVideoAPIView,
    UpdateAnnotionsWithS3URLsAPIView,
)


urlpatterns = [
    # video_frames endpoints
    path("frames/", FramesAPIView.as_view()),
    path("frames-count/", FramesCountAPIView.as_view()),
    path("frames-to-video/", FramesToVideoAPIView.as_view()),
    path("annotations-update/", UpdateAnnotionsWithS3URLsAPIView.as_view()),
]
