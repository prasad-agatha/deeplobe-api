from django.urls import path

from deeplobe_api.api.views.api_key import APIKeyEndpoint, UserAPIKeysDetail

urlpatterns = [
    # api_key endpoints
    path("api-keys/", APIKeyEndpoint.as_view()),
    path("users/api-keys/<int:pk>/", UserAPIKeysDetail.as_view()),
]
