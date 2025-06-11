from django.urls import path

from deeplobe_api.api.views.user import (
    UserCreate,
    UserDetail,
    TaskResults,
    AllUsersListView,
)

urlpatterns = [
    # user endpoints
    path("users/", UserCreate.as_view(), name="create"),
    path("user-details/", UserDetail.as_view(), name="retreive user"),
    path("all-users/", AllUsersListView.as_view(), name="user-list"),
    path("task-results/<str:uuid>/<str:process_type>/", TaskResults.as_view()),
]
