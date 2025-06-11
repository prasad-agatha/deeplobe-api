from django.urls import path

from deeplobe_api.api.views.my_models import MyModels, ModelDetail, ModelDownload

urlpatterns = [
    # my_models endpoints
    path(
        "my-models/",
        MyModels.as_view(),
    ),
    path("model-details/<str:uuid>", ModelDetail.as_view(), name="model detail"),
    path(
        "download-model/<str:uuid>/",
        ModelDownload.as_view(),
    ),
]
