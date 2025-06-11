from django.urls import path

from deeplobe_api.api.views.pretrained_models import (
    PretrainedAPI,
    APILoggerView,
    APIStatistics,
    APILoggerViewAll,
    APIKeyModelsView,
    APIMetricsLogsView,
    APILoggerCustomDetails,
    APIStatisticsByTimePeriodLineChart,
    APIStatisticsByTimePeriodCircularChart,
)


urlpatterns = [
    # pretrained_models endpoints
    path(
        "pretrained-models/<str:kind>",
        PretrainedAPI.as_view(),
        name="Pretrained-models",
    ),
    path(
        "api-logger/",
        APILoggerView.as_view(),
        name="apilogger",
    ),
    path(
        "api-logger-view/<str:type>",
        APILoggerViewAll.as_view(),
        name="apilogger-list",
    ),
    path(
        "api-logger-details/<int:model_id>",
        APILoggerCustomDetails.as_view(),
        name="apilogger-details",
    ),
    path(
        "line_chart_api_statistics_by_time_period/",
        APIStatisticsByTimePeriodLineChart.as_view(),
        name="test3",
    ),
    path(
        "circular_chart_api_statistics_by_time_period/",
        APIStatisticsByTimePeriodCircularChart.as_view(),
        name="test3",
    ),
    path(
        "api_statistics/",
        APIStatistics.as_view(),
        name="api_statistics",
    ),
    path(
        "api_metrics_logs/",
        APIMetricsLogsView.as_view(),
        name="api_metrics_logs",
    ),
    path(
        "api_key_models/",
        APIKeyModelsView.as_view(),
        name="api_key_models",
    ),
]
