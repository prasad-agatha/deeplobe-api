import django_filters

from deeplobe_api.db.models import Country, State


class CountryFilter(django_filters.FilterSet):
    name = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = Country
        fields = ["name"]


class StateFilter(django_filters.FilterSet):
    name = django_filters.CharFilter(field_name="name", lookup_expr="icontains")

    class Meta:
        model = State
        fields = ["name"]
