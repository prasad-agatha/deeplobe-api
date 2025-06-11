from django.urls import path

from deeplobe_api.api.views.cleaner import DataFolderCleanerEndpoint

urlpatterns = [
    # cleaner endpoints
    path("cleaner/", DataFolderCleanerEndpoint.as_view(), name="data-folder-cleaner"),
]
