from django.urls import path

from deeplobe_api.api.views.annotation_hire_request import (
    AnnotationHireExpertRequestView,
    AnnotationHireExpertRequestDetails,
)


urlpatterns = [
    # annotation_hire_request endpoints
    path(
        "annotation-hire-expert/",
        AnnotationHireExpertRequestView.as_view(),
        name="state-detail",
    ),
    path(
        "annotation-hire-expert/<int:pk>",
        AnnotationHireExpertRequestDetails.as_view(),
        name="state-detail",
    ),
]
