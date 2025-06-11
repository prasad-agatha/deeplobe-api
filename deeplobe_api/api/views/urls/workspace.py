from django.urls import path

from deeplobe_api.api.views.workspace import (
    WorkspaceView,
    WorkspaceDetail,
    WorkspaceUsersView,
    PersonalWorkspaceView,
)


urlpatterns = [
    # workspace endpoints
    path(
        "workspace/",
        WorkspaceView.as_view(),
        name="workspace-view",
    ),
    path(
        "workspace/users/",
        WorkspaceUsersView.as_view(),
        name="workspace-users-view",
    ),
    path(
        "workspace/<int:pk>/",
        WorkspaceDetail.as_view(),
        name="workspace-users-view",
    ),
    path(
        "personalworkspace/",
        PersonalWorkspaceView.as_view(),
        name="workspace-users-view",
    ),
]
