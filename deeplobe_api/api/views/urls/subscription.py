from django.urls import path

from deeplobe_api.api.views.subscription import (
    CollaboratorInvitation,
    SubscriberCollaboratorView,
    SubscriberCollaboratorDetail,
    WorkspaceCollaboratorsDeletion,
)

urlpatterns = [
    # subscription endpoints
    path(
        "collaborator-invitation/",
        SubscriberCollaboratorView.as_view(),
        name="collaborator-invitation",
    ),
    path(
        "collaborator-invitation/<int:pk>/",
        SubscriberCollaboratorDetail.as_view(),
        name="collaborator-invitation-detail",
    ),
    path(
        "collaborator-invitation-confirmation/",
        CollaboratorInvitation.as_view(),
        name="collaborator-invitation-confirmation",
    ),
    path(
        "workspace-collaborators-deletion/",
        WorkspaceCollaboratorsDeletion.as_view(),
        name="workspace-collaborators-deletion",
    ),
]
