from django.urls import path

from deeplobe_api.api.views.support import (
    SupportView,
    SupportDetails,
    SupportTicketView,
    SupportTicketListView,
)


urlpatterns = [
    # SupportView endpoints
    path("support/", SupportView.as_view(), name="support"),
    path("support/<int:pk>", SupportDetails.as_view(), name="support"),
    path(
        "support_ticket/<int:ticket_id>",
        SupportTicketView.as_view(),
        name="support-ticket",
    ),
    path(
        "support_ticket_list/",
        SupportTicketListView.as_view(),
        name="support-ticket-list-view",
    ),
]
