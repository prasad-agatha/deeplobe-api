from django.urls import path

from deeplobe_api.api.views.stripe import (
    StripeSubcriptionDetail,
    StripeInvoiceList,
    StripeCreateSession,
    StripeCreatePortalSession,
)

urlpatterns = [
    # state endpoints
    path(
        "stripe/",
        StripeSubcriptionDetail.as_view(),
        name="stripe-detail",
    ),
    path(
        "stripe_invoice_list/",
        StripeInvoiceList.as_view(),
        name="stripe-invoice-list",
    ),
    path(
        "create-session/",
        StripeCreateSession.as_view(),
        name="stripe-create-session",
    ),
    path(
        "create-portal-session/",
        StripeCreatePortalSession.as_view(),
        name="stripe-create-portal-session",
    ),
]
