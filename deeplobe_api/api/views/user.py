import os
import base64
import stripe
import chargebee

from decouple import config

from django.http import Http404
from django.core.files import File

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from deeplobe_api.db.models import StripeSubcription

from deeplobe_api.db.models import (
    User,
    Task,
    Subscription,
    Workspace,
    UserInvitation,
)
from deeplobe_api.api.serializers import (
    UserSerializer,
    TaskSerializer,
    WorkspaceSerializer,
)

from deeplobe_api.api.views.auth import create_charge_account
from deeplobe_api.api.views.permissions import DomainValidationPermissions
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails


chargebee.configure(
    site=config("CHARGEBEE_SITE_KEY"), api_key=config("CHARGEBEE_API_ACCESS_KEY")
)

stripe.api_key = config("STRIP_API_KEY")

class UserCreate(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserDetail(APIView):
    """
    Retrieve, delete a user
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        """
        Return user object if pk value present.
        """
        try:
            return User.objects.get(pk=pk)
        except User.DoesNotExist:
            raise Http404

    def get(self, request, format=None):
        """
        Return user.
        """
        user = self.get_object(request.user.id)
        serializer = UserSerializer(user)
        result = serializer.data
        all_workspaces = []
        personal_workspace = Workspace.objects.filter(user=request.user).first()
        current_workspace = Workspace.objects.filter(
            id=request.user.current_workspace
        ).first()
        if personal_workspace:
            all_workspaces.append(personal_workspace)
        else:
            name = user.username + "'s Team"
            workspace = Workspace.objects.create(name=name, user=user)
            user.current_workspace = workspace.id
            user.save()
            # customer_id, subscription_id = create_charge_account(
            #     user.username, request.user.email
            # )
            # Subscription.objects.create(
            #     customer_id=customer_id,
            #     subscription_id=subscription_id,
            #     subscriber=user,
            # )
            personal_workspace = Workspace.objects.filter(user=request.user).first()
            all_workspaces.append(personal_workspace)
        for collaborator in UserInvitation.objects.filter(
            collaborator_email=request.user.email, is_collaborator=True
        ):
            all_workspaces.append(collaborator.workspace)
        result["current_workspace_details"] = chargebeeplandetails(
            email=current_workspace.user
        )
        personal_workspace_serializer = WorkspaceSerializer(personal_workspace)
        if personal_workspace_serializer.data["id"] == request.user.current_workspace:
            result["current_workspace_details"]["role"] = "owner"
            result["current_workspace_details"]["models"] = [
                {
                    "select": ["All"],
                    "api": True,
                    "create_model": True,
                    "delete_model": True,
                    "test_model": True,
                }
            ]
        else:
            collaborator_user = UserInvitation.objects.filter(
                workspace=request.user.current_workspace,
                collaborator_email=request.user.email,
                is_collaborator=True,
            ).first()
            if collaborator_user:
                result["current_workspace_details"]["role"] = collaborator_user.role
                result["current_workspace_details"]["models"] = collaborator_user.models
            else:
                result["current_workspace_details"]["role"] = "None"
                result["current_workspace_details"]["models"] = []

        result["workspaces"] = WorkspaceSerializer(all_workspaces, many=True).data

        for work in result["workspaces"]:
            workspace_plan_details = chargebeeplandetails(
                email=User.objects.filter(email=work["user"]).first()
            )
            if workspace_plan_details:
                work["plan"] = workspace_plan_details["plan"]
            else:
                work["plan"] = ""

        return Response(result, status=status.HTTP_200_OK)

    def put(self, request, format=None):
        user = self.get_object(request.user.id)
        mutable_request_data = request.data.copy()
        if request.data.get("postal_code")=="":
            mutable_request_data["postal_code"]=None
        if request.data.get("current_workspace"):
            personal_workspace = Workspace.objects.filter(user=request.user).first()
            personal_workspace_serializer = WorkspaceSerializer(personal_workspace)
            if (
                personal_workspace_serializer.data["id"]
                != request.data.get("current_workspace")
                and not UserInvitation.objects.filter(
                    workspace=request.data.get("current_workspace"),
                    collaborator_email=request.user.email,
                    is_collaborator=True,
                ).first()
            ):
                return Response(
                    {"msg": "User not in workspace"}, status=status.HTTP_400_BAD_REQUEST
                )
        if (
            (request.data.get("address") != None and 
            request.data.get("address") != request.user.address)
            or (request.data.get("country")!=None and request.data.get("country") != request.user.country)
            or (request.data.get("state")!=None and request.data.get("state") != request.user.state)
            or (request.data.get("postal_code")!=None and request.data.get("postal_code") != request.user.postal_code)
            or (request.data.get("contact_number")!=None and request.data.get("contact_number") != request.user.contact_number)
          
        ):
            if request.user.stripe_customer_id:
                stripe.Customer.modify(
                    request.user.stripe_customer_id,
                    address={
                        "country": request.data.get("country"),
                        "line1": request.data.get("address"),
                        "postal_code": request.data.get("postal_code"),
                        "state": request.data.get("state"),
                    },
                    phone=request.data.get("contact_number"),
                )
            if not request.user.stripe_subcription and (request.data.get("country")!=None and request.data.get("country") != request.user.country):
                mutable_request_data["stripe_currency"] = "usd" if request.data.get("country")!= "IN" else "inr"

            serializer = UserSerializer(user, data=mutable_request_data, partial=True)
        else:
            serializer = UserSerializer(user, data=request.data, partial=True)

        # user.is_new = False
        user.save()

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


    def delete(self, request, format=None):
        """
        Delete user.
        """
        user = self.get_object(request.user.id)
        subscriptions = StripeSubcription.objects.filter(
           customer_email=request.user.email

        )
        for subscription in subscriptions:
            subscription.delete()
        user_as_collaborator = UserInvitation.objects.filter( collaborator_email=request.user.email)
        for collaborator in user_as_collaborator:
            collaborator.delete()
        user.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


def image_output(path):
    if os.path.isfile(os.path.abspath(path)):
        f = open(os.path.abspath(path), "rb")
        image = File(f)
        data = base64.b64encode(image.read())
        f.close()

        return {
            "status": 1,
            "info": "Image exist",
            "base64_image": data.decode("utf-8"),
        }

    else:
        return {"status": 0, "info": "Image not exist"}


class TaskResults(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, uuid, process_type):
        try:
            obj = Task.objects.get(uuid=uuid, task_type=process_type)

            serializer = TaskSerializer(obj, many=False)
            data = serializer.data
            data["signal"] = 1
            data["msg"] = "record present"
            return Response(data, status=status.HTTP_200_OK)

        except Exception as e:
            print(e)
            return Response(
                {"signal": 0, "msg": "record still not present"},
                status=status.HTTP_200_OK,
            )


class UserProfilePicUpdate(APIView):
    def get_object(self, pk):
        try:
            return User.objects.get(pk=pk)
        except User.DoesNotExist:
            raise Http404

    def put(self, request, pk):
        user = self.get_object(pk)
        serializer = UserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AllUsersListView(APIView):
    permission_classes = (
        IsAuthenticated,
        DomainValidationPermissions,
    )

    def get(self, request):
        queryset = User.objects.all()
        serializer = UserSerializer(queryset, many=True)
        lst = list(serializer.data)
        result = []
        for user in lst:
            data = dict(user)
            workspace_plan_details = chargebeeplandetails(
                email=User.objects.filter(email=user["email"]).first()
            )
            if workspace_plan_details:
                data["plan"] = workspace_plan_details["plan"]
                data["plan_status"] = workspace_plan_details["plan_status"]
            else:
                data["plan"] = ""
                data["plan_status"] = ""
            result.append(data)

        return Response(result, status=status.HTTP_200_OK)
