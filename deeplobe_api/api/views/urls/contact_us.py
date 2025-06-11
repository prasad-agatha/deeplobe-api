from django.urls import path

from deeplobe_api.api.views.contact_us import ContactUsViewSet, ContactUsDetails


urlpatterns = [
    # contact_us endpoints
    path(
        "contact-us/",
        ContactUsViewSet.as_view(),
    ),
    path(
        "contact-us/<int:pk>",
        ContactUsDetails.as_view(),
    ),
]
