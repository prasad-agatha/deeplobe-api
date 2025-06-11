from django.urls import path

from deeplobe_api.api.views.media_asset import (
    MediaAssetView,
    MediaAssetDetailView,
    ImageResizeEndpoint,
    ImageResize,
)


urlpatterns = [
    # media_asset endpoints
    path("assets/", MediaAssetView.as_view(), name="assets"),
    path("assets/resize/<str:uuid>", ImageResize.as_view(), name="resize"),
    path("assets/<int:pk>/", MediaAssetDetailView.as_view(), name="asset-detail"),
    path("assets/<int:pk>/resize/", ImageResizeEndpoint.as_view(), name="image-resize"),
]
