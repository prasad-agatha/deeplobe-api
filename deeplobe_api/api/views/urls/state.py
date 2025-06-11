from django.urls import path

from deeplobe_api.api.views.state import StateList, StateDetail

urlpatterns = [
    # state endpoints
    path(
        "state/",
        StateList.as_view(),
        name="state-list",
    ),
    path(
        "state/<int:pk>",
        StateDetail.as_view(),
        name="state-detail",
    ),
]
