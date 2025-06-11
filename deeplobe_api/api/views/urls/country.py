from django.urls import path

from deeplobe_api.api.views.country import CountryList, CountryDetail

urlpatterns = [
    # country endpoints
    path(
        "country/",
        CountryList.as_view(),
        name="country-list",
    ),
    path(
        "country/<int:pk>",
        CountryDetail.as_view(),
        name="country-detail",
    ),
]
