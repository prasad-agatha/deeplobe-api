from django.urls import path

from deeplobe_api.api.views.register import RegisterList, RegisterDetail


urlpatterns = [
    # register endpoints
    path("registerlist/", RegisterList.as_view()),
    path("registerdetail/<int:pk>", RegisterDetail.as_view()),
]
