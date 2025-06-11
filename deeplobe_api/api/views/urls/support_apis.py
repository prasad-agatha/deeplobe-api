from django.urls import path

from deeplobe_api.api.views.support_apis import UuidStatus, AllCategories, CheckName

urlpatterns = [
    # support_apis endpoints
    path("all-categories/", AllCategories.as_view()),
    path("uuid-status/<str:uuid>/", UuidStatus.as_view()),
    path(
        "check-name/<str:name>/",
        CheckName.as_view(),
    ),
]
