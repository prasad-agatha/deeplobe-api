import time
import stripe

from decouple import config

from django.shortcuts import get_object_or_404


from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from deeplobe_api.db.models import StripeSubcription, SubcriptionInvoices
from deeplobe_api.api.serializers import StripeSerializer, SubscriptionInvoiceSerializer
from deeplobe_api.db.models import User
from deeplobe_api.utils.emails import emails


stripe.api_key = config("STRIP_API_KEY")


class StripeSubcriptionDetail(APIView):
    def post(self, request):
        event_type = request.data["event"]["type"]
        event_data = request.data["event"]["data"]["object"]
        stripe_customer_id = (
            event_data["id"]
            if event_type == "customer.updated"
            else event_data["customer"]
        )

        user = get_object_or_404(User, stripe_customer_id=stripe_customer_id)

        if event_type == "customer.updated":
            # billing address
            user.username = event_data["name"]
            user.address = event_data["address"]["line1"]
            user.country = event_data["address"]["country"]
            user.state = event_data["address"]["state"]
            user.postal_code = event_data["address"]["postal_code"]
            user.stripe_customer_id = stripe_customer_id
            if event_data["currency"]:
                user.stripe_currency = event_data["currency"]
            user.save()
            return Response("Success", status=status.HTTP_200_OK)
        else:
            user_subscription_id = (
                event_data["subscription"]
                if event_type == "invoice.paid"
                else event_data["id"]
            )
            user_subscription, is_created = StripeSubcription.objects.get_or_create(
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=user_subscription_id,
            )
            if is_created:
                user.stripe_subcription = user_subscription
                user.save()
            if event_type == "invoice.paid":
                # Invoice Details with PDF
                invoice = event_data
                no_plan = {"nickname": "", "created": "", "currency": "", "amount": 0}
                plan = (
                    invoice["lines"]["data"][0]["plan"]
                    if len(invoice["lines"]["data"]) > 0
                    else no_plan
                )
                invoice_details = {
                    "id": invoice["id"],
                    "invoice_pdf": invoice["invoice_pdf"],
                    "plan": plan["nickname"],
                    "amount": str(plan["currency"]).upper()
                    + " "
                    + str(plan["amount"] / 100),
                    "date": str(
                        time.strftime("%Y-%m-%d", time.localtime(invoice["created"]))
                    )
                    if invoice["created"]
                    else "",
                    # Include other relevant invoice details as needed
                }

                SubcriptionInvoices.objects.create(
                    subscription=user_subscription, invoice_details=invoice_details
                )
                return Response("Success", status=status.HTTP_200_OK)

            elif event_type in [
                "customer.subscription.created",
                "customer.subscription.updated",
                "customer.subscription.deleted",
            ]:
                price = event_data["items"]["data"][0]["price"]

                user_subscription.price = price
                user_subscription.metadata = price["metadata"]
                user_subscription.stripe_subscription = event_data
                user_subscription.status = event_data["status"]
                user_subscription.plan_name = price["nickname"]
                user_subscription.quantity = event_data["quantity"]
                user_subscription.canceled_at = (
                    str(
                        time.strftime(
                            "%Y-%m-%d", time.localtime(event_data["cancel_at"])
                        )
                    )
                    if event_data["cancel_at"]
                    else ""
                )
                user_subscription.current_period_start = (
                    str(
                        time.strftime(
                            "%Y-%m-%d",
                            time.localtime(event_data["current_period_start"]),
                        )
                    )
                    if event_data["current_period_start"]
                    else ""
                )
                user_subscription.current_period_end = (
                    str(
                        time.strftime(
                            "%Y-%m-%d",
                            time.localtime(event_data["current_period_end"]),
                        )
                    )
                    if event_data["current_period_end"]
                    else ""
                )
                user_subscription.created = (
                    str(
                        time.strftime(
                            "%Y-%m-%d",
                            time.localtime(event_data["created"]),
                        )
                    )
                    if event_data["created"]
                    else ""
                )
                user_subscription.ended_at = (
                    str(
                        time.strftime(
                            "%Y-%m-%d",
                            time.localtime(event_data["ended_at"]),
                        )
                    )
                    if event_data["ended_at"]
                    else ""
                )

                # Get invoice details
                invoice_id = event_data["latest_invoice"]
                invoice = stripe.Invoice.retrieve(invoice_id)
                user_subscription.customer_email = invoice["customer_email"]
                user_subscription.save()

                # Send email notification
                if event_type == "customer.subscription.created":
                   emails.stripe_subscription_created_email(request, user)
                elif event_type == "customer.subscription.updated":
                     emails.stripe_subscription_plan_change_email(request, user,user_subscription)
                elif event_type == "customer.subscription.deleted":
                     emails.stripe_subscription_plan_delete_email(request, user,user_subscription)


                return Response("Success", status=status.HTTP_200_OK)

            return Response("Invalid Event", status=status.HTTP_400_BAD_REQUEST)

class StripeInvoiceList(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = SubcriptionInvoices.objects.filter(
            subscription__customer_email=request.user.email
        )
        serializer = SubscriptionInvoiceSerializer(queryset, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)


class StripeCreateSession(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        customer_id = request.user.stripe_customer_id
        if not customer_id:
            customer = stripe.Customer.create(
                email=request.user.email,
                name=request.user.username,
                address={
                    "city": "",
                    "country": request.user.country,
                    "line1": request.user.address,
                    "line2": "",
                    "postal_code": request.user.postal_code,
                    "state": request.user.state,
                },
                phone=request.user.contact_number,
            )
            user = get_object_or_404(User, email=request.user.email)
            user.stripe_customer_id = customer.id
            user.save()
            customer_id = customer.id
        if request.data.get("period") == "Half-Yearly":
            if request.user.stripe_currency == "inr":
                price = config("STRIPE_GROWTH_HALF_YEARLY_INR")
            else:
                price = config("STRIPE_GROWTH_HALF_YEARLY")
        else:
            if request.user.stripe_currency == "inr":
                price = config("STRIPE_GROWTH_YEARLY_INR")
            else:
                price = config("STRIPE_GROWTH_YEARLY")
            
        response = stripe.checkout.Session.create(
            line_items=[
                {
                    "price": price,
                    "quantity": 1,
                },
            ],
            
            subscription_data={
                    'default_tax_rates': [config("STRIPE_TAX_INR")] if request.user.stripe_currency == "inr" else [] ,
                },
            mode="subscription",
            customer=customer_id,
            payment_method_types=["card"],
            allow_promotion_codes=True,
            success_url=f"{request.headers['Origin']}/settings?success=true",
            cancel_url=f"{request.headers['Origin']}/settings?cancelled=true",
        )
        return Response({"url": response.url}, status=status.HTTP_200_OK)


class StripeCreatePortalSession(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        customer_id = request.user.stripe_customer_id
        if not customer_id:
            return Response("User not found", status=status.HTTP_404_NOT_FOUND)
        response = stripe.billing_portal.Session.create(
            customer=request.user.stripe_customer_id,
            return_url=f"{request.headers['Origin']}/settings",
        )
        return Response({"url": response.url}, status=status.HTTP_200_OK)
