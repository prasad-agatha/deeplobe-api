from django.urls import path, include


urlpatterns = [
    # django-rq endpoints
    path("django-rq/", include("django_rq.urls")),
]
