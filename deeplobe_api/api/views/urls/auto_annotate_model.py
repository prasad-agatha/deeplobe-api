from django.urls import path

from deeplobe_api.api.views.auto_annotate_model import (
    AutoAnnotateListView,
    AutoAnnotateDetailView,
)


urlpatterns = [
    # auto_annotate_model endpoints
    path(
        "autoannotate/",
        AutoAnnotateListView.as_view(),
        name="auto-annotate-model",
    ),
    path(
        "autoannotate/<int:pk>",
        AutoAnnotateDetailView.as_view(),
        name="auto-annotate-model-details",
    ),
]
