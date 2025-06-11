import time
import chargebee

from decouple import config

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from deeplobe_api.db.models import Subscription


from deeplobe_api.api.views.function_calls.subscription import deletecollaborators

chargebee.configure(
    site=config("CHARGEBEE_SITE_KEY"), api_key=config("CHARGEBEE_API_ACCESS_KEY")
)


class ChargeBeeSubscriptionUpdate(APIView):

    """
    API endpoint that allows user to update subscription.

    * Requires JWT authentication.
    * This endpoint will allows only PUT methods.
    """

    permission_classes = [IsAuthenticated]

    def put(self, request):
        """
        Update subscription plan .
        """
        subscription = Subscription.objects.filter(subscriber=request.user).first()
        subscription_i = subscription.subscription_id
        result = chargebee.Subscription.update_for_items(
            subscription_i,
            {
                "subscription_items": [
                    {"item_price_id": "Growth-INR-Monthly", "item_type": "plan"}
                ]
            },
        )
        subscription = result.subscription


class ChargeBeeBillingInfo(APIView):

    """
    API endpoint that allows user to create chargebee_billing_info or view the list of allusers chargebee_billing_info.

    * Requires JWT authentication.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of all chargebee_billing_info.
        """

        subscription = Subscription.objects.filter(subscriber=request.user).first()
        customer_id = subscription.customer_id
        customer_info = chargebee.Customer.retrieve(customer_id)
        customer_info = customer_info.__dict__
        billing_info = customer_info["_response"]["customer"].get("billing_address")
        return Response({"billing_info": billing_info}, status=status.HTTP_200_OK)

    def post(self, request):
        """
        Create a new chargebee_billing_info.
        """
        subscription = Subscription.objects.filter(subscriber=request.user).first()
        customer_id = subscription.customer_id
        billing_info = chargebee.Customer.update_billing_info(
            customer_id,
            {
                "billing_address": {
                    "first_name": request.data.get("first_name"),
                    "last_name": request.data.get("last_name"),
                    "address_line": request.data.get("address_line"),
                    "city": request.data.get("city"),
                    "state": request.data.get("state"),
                    "zip": request.data.get("zip"),
                    "country": request.data.get("country"),
                    # "vat_number": request.data.get("vat_number"),
                }
            },
        )
        billing_address = billing_info.__dict__
        billing_address_info = billing_address["_response"]["customer"][
            "billing_address"
        ]
        return Response(billing_address_info, status=status.HTTP_200_OK)


class ChargeBeeInvoiceList(APIView):

    """
    API endpoint that allows user to  view the list of all users chargebee_invoice.

    * Requires JWT authentication.
    * This endpoint will allows only GET methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of all chargebee_invoice.
        """
        subscription = Subscription.objects.filter(subscriber=request.user).first()
        customer_id = subscription.customer_id
        entries = chargebee.Invoice.list(
            {"customer_id[is]": customer_id, "sort_by[desc]": "date"}
        )
        invoices = []
        for entry in entries:
            result = entry.invoice
            invoice = {}
            invoice["id"] = result.id
            if len(result.line_items) > 0:
                invoice["plan"] = result.line_items[0].entity_id
            else:
                invoice["plan"] = ""
            invoice["amount"] = result.currency_code + " " + str(result.total / 100)
            invoice["date"] = time.strftime("%Y-%m-%d", time.localtime(result.date))
            invoices.append(invoice)
        return Response(invoices, status=status.HTTP_200_OK)


class ChargeBeeInvoicePDFDownload(APIView):

    """
    API endpoint that allows user to download chargebee_invoice_pdf.

    * Requires JWT authentication.
    * This endpoint will allows only POST methods.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        """
        download chargebee_invoice_pdf.
        """

        invoice_pdf = chargebee.Invoice.pdf(request.data.get("id"))
        downloads_pdf = invoice_pdf.download.download_url
        return Response({"url": downloads_pdf}, status=status.HTTP_200_OK)


class ChargeBeeSession(APIView):

    """
    API endpoint that allows user to view chargebee_portal_session.

    * Requires JWT authentication.
    * This endpoint will allows only GET methods.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return chargebee_portal_session.
        """
        subscription = Subscription.objects.filter(subscriber=request.user).first()
        customer_id = subscription.customer_id

        result = chargebee.PortalSession.create(
            {
                "customer": {"id": customer_id},
            }
        )
        portal_session = result.portal_session.__dict__
        del portal_session["linked_customers"]
        del portal_session["sub_types"]
        return Response(portal_session, status=status.HTTP_200_OK)


class ChargeBeeSubscriptionCancellation(APIView):

    """
    API endpoint that allows user to cancel chargbee_subscription.

    * Requires JWT authentication.
    * This endpoint will allows only POST methods.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        """
        Cancel chargbee_subscription.
        """

        subscription = Subscription.objects.filter(subscriber=request.user).first()
        customer_id = subscription.customer_id
        entries = chargebee.Subscription.list({"customer_id[is]": customer_id})
        for entry in entries:
            subscription = entry.subscription
        if (
            subscription
            and subscription.status
            and len(subscription.subscription_items) > 0
            and subscription.subscription_items[0].item_price_id
            != "DeepLobe-Free-Plan-USD-Monthly"
        ):
            chargebee.Subscription.update_for_items(
                subscription.id,
                {
                    "subscription_items": [
                        {
                            "item_price_id": "DeepLobe-Free-Plan-USD-Monthly",
                            "item_type": "plan",
                        }
                    ]
                },
            )
            deletecollaborators(request)
            return Response(
                "Subscription Cancelled successfully", status=status.HTTP_200_OK
            )

        return Response(
            "Error cancelling subscription", status=status.HTTP_400_BAD_REQUEST
        )
