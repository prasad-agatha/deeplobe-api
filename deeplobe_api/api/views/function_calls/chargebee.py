import uuid
import time
import chargebee

from decouple import config

from rest_framework import status
from rest_framework.response import Response

from django.db.models import Count, Max
from deeplobe_api.db.models import (
    User,
    AIModel,
    APICount,
    APILogger,
    Workspace,
    Subscription,
    UserInvitation,
)
from deeplobe_api.api.serializers import UserSerializer
from deeplobe_api.db.models import StripeSubcription
from deeplobe_api.api.serializers import StripeSerializer
from datetime import datetime, timedelta,date

def create_charge_account(name, email):
    first_name, last_name = name.split()[0], " ".join(name.split()[1:])
    chargebee_customer = chargebee.Customer.create(
        {
            "id": uuid.uuid4().hex,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "company": config("CLIENT_NAME"),
            "billing_address": {
                "first_name": first_name,
                "last_name": last_name,
                "state": "Telangana",
                "country": "IN",
            },
        }
    )
    customer = chargebee_customer.customer
    if customer:
        # create subscription for customer in chargebee
        chargebee_subscription = chargebee.Subscription.create_with_items(
            customer.id,
            {
                "subscription_items": [
                    {
                        "item_price_id": "DeepLobe-Free-Plan-USD-Monthly",
                        "item_type": "plan",
                    }
                ]
            },
        )
        subscription = chargebee_subscription.subscription
        return customer.id, subscription.id
    else:
        print("customer not exists")




def chargebeeplandetails(email):
    try:
        current_date = datetime.now().date()
        current_date_str = datetime.now()
        user_obj = User.objects.filter(email=email).first()
        serializer = UserSerializer(user_obj)
        user = serializer.data
        subcription = None
        subscription_period = ""
        if user["stripe_subcription"]:
            subcription_obj = StripeSubcription.objects.filter(
                id=user["stripe_subcription"]
            ).first()
            subcription_serializer = StripeSerializer(subcription_obj)
            subcription = subcription_serializer.data
            # TODO
            subscription_period = datetime.strptime(subcription["current_period_end"] + " 0:00:00.0", '%Y-%m-%d %H:%M:%S.%f')
        if subcription and subcription["status"] == "active" and subscription_period>=current_date_str:
            meta_data = {
                "api-requests": 10000,
                "custom-models": True,
                "storage": "1TB",
                "users": 3,
                "from_date": subcription["current_period_start"],
                "to_date": subcription["current_period_end"],
                "plan_status": subcription["status"],
                "plan": subcription["plan_name"],
                "billing_period": "",
            }
        else:
            meta_data = { 
                "api-requests": 1000,
                "custom-models": False,
                "storage": "10GB",
                "users": 1,
                "from_date": str(current_date - timedelta(days=1))if not subcription else subcription["current_period_start"],
                "to_date": str(current_date + timedelta(days=1)) if not subcription else subcription["current_period_end"],
                "plan_status": "active",
                "plan": "Free-plan",
                "billing_period": "",
            }
         
        workspace = Workspace.objects.filter(user=user_obj).first()
        collaborators = UserInvitation.objects.filter(
            workspace=workspace, is_collaborator=True, role="collaborator"
        ).count()
        qs = (
            APILogger.objects.filter(workspace=workspace, model_type__startswith="pre")
            .values("model_type")
            .annotate(model_type_count=Count("model_type"), created=Max("created"))
        )
        pretrained_count = qs.count()
        custom_model_count = AIModel.objects.filter(
            workspace=workspace, is_trained=True, is_active=True,status="Live"
        ).count()
        apicount = APICount.objects.filter(
            workspace=workspace.id,
        )
        if apicount.exists():
            apicalls = (
                apicount[0].custom_model_api_count
                + APILogger.objects.filter(api_key__user=user_obj).count()
            )

        else:
            apicalls = APILogger.objects.filter(api_key__user=user_obj).count()
        meta_data["api_calls"] = apicalls
        meta_data["collaborators"] = collaborators
        if meta_data["plan"] == "Free-plan":
            meta_data["active_models"] = pretrained_count
        else:
            meta_data["active_models"] = pretrained_count + custom_model_count
        return meta_data

    except Exception as e:
        return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
