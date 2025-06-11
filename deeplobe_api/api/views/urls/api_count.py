from django.urls import path

from deeplobe_api.api.views.api_count import APICountList, APICountDetail


urlpatterns = [
    # api_count endpoints
    path(
        "apicount/",
        APICountList.as_view(),
        name="api-restriction-details",
    ),
    path(
        "apicount/<int:pk>",
        APICountDetail.as_view(),
        name="api-restriction-details",
    ),
]
