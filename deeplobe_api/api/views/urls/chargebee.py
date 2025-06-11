from django.urls import path

from deeplobe_api.api.views.chargebee import (
    ChargeBeeSession,
    ChargeBeeBillingInfo,
    ChargeBeeInvoiceList,
    ChargeBeeInvoicePDFDownload,
    ChargeBeeSubscriptionUpdate,
    ChargeBeeSubscriptionCancellation,
)


urlpatterns = [
    # chargebee endpoints
    path(
        "chargebeeupdate/",
        ChargeBeeSubscriptionUpdate.as_view(),
        name="workspace-users-view",
    ),
    path(
        "chargebeebillinginfo/",
        ChargeBeeBillingInfo.as_view(),
        name="chargebee-billing-info",
    ),
    path(
        "chargebeeinvoicelist/",
        ChargeBeeInvoiceList.as_view(),
        name="chargebee-invoice-list",
    ),
    path(
        "chargebeeinvoicepdfgenerator/",
        ChargeBeeInvoicePDFDownload.as_view(),
        name="chargebee-invoice-pdf-generator",
    ),
    path(
        "chargebeesession/",
        ChargeBeeSession.as_view(),
        name="chargebee-session",
    ),
    path(
        "chargebeesubscriptioncancellation/",
        ChargeBeeSubscriptionCancellation.as_view(),
        name="chargebee_subscription_cancellation",
    ),
]
