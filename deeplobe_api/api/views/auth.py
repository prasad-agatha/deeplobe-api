import uuid
import chargebee


from decouple import config

from sesame.utils import get_token, get_user

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_jwt.settings import api_settings
from rest_framework.permissions import IsAuthenticated, IsAdminUser

from django.http import Http404
from django.utils import timezone
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from deeplobe_api.db.models import (
    User,
    SocialProvider,
    Subscription,
    Workspace,
)
from deeplobe_api.db.models import StripeSubcription
from deeplobe_api.utils.emails import emails
from deeplobe_api.api.views.permissions import AdminIdentity
from deeplobe_api.api.views.function_calls.chargebee import create_charge_account
from deeplobe_api.api.views.function_calls.subscription import new_user_invitations
from deeplobe_api.api.views.function_calls.auth import payload_generate, google_verify
from deeplobe_api.api.serializers import RegisterSerializer, UserSerializer


jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER


class LoginDetailMixin:
    def user_data(self, user, token):
        user_data = {}
        user_data["user_id"] = user.id
        user_data["email"] = user.email
        user_data["user_first_name"] = user.first_name
        user_data["user_last_name"] = user.last_name
        user_data["username"] = user.username
        user_data["industry"] = user.industry
        user_data["role"] = user.role
        user_data["company"] = user.company
        user_data["contact_number"] = user.contact_number
        user_data["user_is_staff"] = user.is_staff
        user_data["user_is_active"] = user.is_active
        user_data["user_is_superuser"] = user.is_superuser
        user_data["token"] = token
        user_data["last_login"] = user.last_login
        user_data["is_new"] = user.is_new
        user_data["model_permissions"] = user.model_permissions
        user_data["current_workspace"] = user.current_workspace
        user_data["postal_code"] = user.postal_code
        user_data["state"] = user.state
        user_data["country"] = user.country
        user_data["GSTIN"] = user.GSTIN
        user_data["company_size"] = user.company_size
        user_data["address"] = user.address
        user_data["bussiness_email"] = user.bussiness_email
        user_data["help"] = user.help
        user_data["notes"] = user.notes

        return user_data

    def social_provider_auth(self, user, provider, extra):
        provider, created = SocialProvider.objects.get_or_create(
            user=user, provider=provider
        )
        provider.extra = extra
        provider.save()
        return


class SocialAuthView(LoginDetailMixin, APIView):
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super(SocialAuthView, self).dispatch(request, *args, **kwargs)

    def post(self, request):
        """
        Generate JWT Token
        """
        return_data = google_verify(request)
        if ("error" in return_data) or ("error_description" in return_data):
            return Response(return_data, status=status.HTTP_400_BAD_REQUEST)
        email = return_data.get("email", None)
        first_name = return_data.get("family_name", None)
        last_name = return_data.get("given_name", None)
        model_permissions = return_data.get("model_permissions", None)
        postal_code = return_data.get("postal_code", None)
        state = return_data.get("state", None)
        country = return_data.get("country", None)
        GSTIN = return_data.get("GSTIN", None)
        company_size = return_data.get("company_size", None)
        address = return_data.get("address", None)
        bussiness_email = return_data.get("bussiness_email", None)
        provider = "Google"
        extra = return_data
        firstname, at, company = email.rpartition("@")
        # companies = ["intellectdata.com", "soulpageit.com"]

        # model_permissions = [
        #     {
        #         "pretrained": True if company in companies else False,
        #         "image_classification": True if company in companies else False,
        #         "image_similarity": True if company in companies else False,
        #         "object_detection": True if company in companies else False,
        #         "semantic_segmentation": True if company in companies else False,
        #         "instance_segmentation": True if company in companies else False,
        #         "optical_character_recognition": True
        #         if company in companies
        #         else False,
        #     }
        # ]
        model_permissions = [
            {
                "pretrained": True,
                "image_classification": True,
                "image_similarity": True,
                "object_detection": True,
                "semantic_segmentation": True,
                "instance_segmentation": True,
                "optical_character_recognition": True,
                "llm":True,
            }
        ]

        if email == None:
            return Response(
                {"error": "Missing fields"}, status=status.HTTP_400_BAD_REQUEST
            )
        try:
            user = User.objects.get(email=email)

            data = payload_generate(self, user)
            # Update Social Provider Details
            self.social_provider_auth(user, provider, extra)
            return Response(data, status=status.HTTP_200_OK)

        except:
            pw = str(uuid.uuid4())
            user = User.objects.create_user(
                email=email,
                password=pw,
                first_name=first_name,
                last_name=last_name,
                company="",
                terms_conditions=True,
                username=first_name + " " + last_name,
                model_permissions=model_permissions,
                postal_code=postal_code,
                state=state if state else "TG",
                country=country if country else "IN",
                GSTIN=GSTIN,
                company_size=company_size,
                address=address,
                bussiness_email=bussiness_email,
                stripe_currency="inr"
            )
            try:
                # add workspace to user
                name = user.username + "'s Team"
                workspace = Workspace.objects.create(name=name, user=user)
                user.current_workspace = workspace.id
                user.save()
                # create chargebee account
                # customer_id, subscription_id = create_charge_account(
                #     user.username, email
                # )

                # Subscription.objects.create(
                #     customer_id=customer_id,
                #     subscription_id=subscription_id,
                #     subscriber=user,
                # )
                new_user_invitations(user, email)

            except Exception as e:
                print(str(e))
            data = payload_generate(self, user)
            self.social_provider_auth(user, provider, extra)
            return Response(data, status=status.HTTP_201_CREATED)


class Sendmagiclink(APIView):
    def post(self, request):
        email = request.data.get("email", None)
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {
                    "email": "Email not found",
                },
                status=400,
            )

        # serializer = UserSerializer(user)
        # if not user.is_active:
        #     return Response(
        #         {
        #             "msg": "User is inactive please contact to admin",
        #             "user": serializer.data,
        #         }
        #     )
        token = get_token(user)
        emails.send_magic_link(email, user, token)
        return Response({"msg": "magic link send succesfully"})


class VerifyMagicLink(APIView):
    def payload_generate_token(self, user):
        payload = jwt_payload_handler(user)

        token = jwt_encode_handler(payload)

        return token

    def get(self, request):
        try:
            magic_token = request.query_params.get("token")
            user = get_user(magic_token)
            serializer = UserSerializer(user)
            token = self.payload_generate_token(user)
            return Response({"token": token, "user": serializer.data})
        except:
            return Response(
                {"msg": "send valid token"},
                status=status.HTTP_400_BAD_REQUEST,
            )


# class UpdateFirstTimeUser(APIView):

#     permission_classes = [IsAuthenticated]

#     def get_object(self, pk):
#         """
#         Return user object if pk value present.
#         """
#         try:
#             return User.objects.get(pk=pk)
#         except User.DoesNotExist:
#             raise Http404

#     def put(self, request):

#         try:
#             user = self.get_object(request.user.id)
#             serializer = UserSerializer(user, data=request.data, partial=True)
#             user.is_new = False
#             user.save()
#             if serializer.is_valid():
#                 serializer.save()

#                 return Response(
#                     {"success": True, "user": serializer.data},
#                     status=status.HTTP_201_CREATED,
#                 )

#         except:
#             return Response(
#                 {"success": False},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )


class MagicLinkRegister(APIView):
    def post(self, request):
        try:
            email = request.data.get("email", None)
            firstname, at, company = email.rpartition("@")
            # companies = [
            #     "intellectdata.com",
            #     "soulpageit.com",
            # ]
            # if company in companies:
            model_permissions = [
                {
                    "pretrained": True,
                    "image_classification": True,
                    "image_similarity": True,
                    "object_detection": True,
                    "semantic_segmentation": True,
                    "instance_segmentation": True,
                    "optical_character_recognition": True,
                    "llm":True,
                }
            ]
            # else:
            #     model_permissions = [
            #         {
            #             "pretrained": False,
            #             "image_classification": False,
            #             "image_similarity": False,
            #             "object_detection": False,
            #             "semantic_segmentation": False,
            #             "instance_segmentation": False,
            #             "optical_character_recognition": False,
            #         }
            #     ]
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            user = User.objects.create(
                username=firstname,
                email=email,
                model_permissions=model_permissions,
                state="TG",
                country="IN",
                stripe_currency="inr"
            )
            try:
                # add workspace to user
                name = user.username + "'s Team"
                workspace = Workspace.objects.create(name=name, user=user)
                user.current_workspace = workspace.id
                user.save()
                # create chargebee account
                # customer_id, subscription_id = create_charge_account(
                #     user.username, email
                # )

                # Subscription.objects.create(
                #     customer_id=customer_id,
                #     subscription_id=subscription_id,
                #     subscriber=user,
                # )
                new_user_invitations(user, email)

            except Exception as e:
                print(str(e))
        serializer = RegisterSerializer(
            data={
                "email": email,
            }
        )

        if serializer.is_valid(raise_exception=True):
            serializer.save()
            token = get_token(user)
            emails.magic_link_register(email, user, token)
            return Response(
                {"msg": "user created successfully"},
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ActivateUserEndpoint(APIView):
    permission_classes = (IsAdminUser,)

    def get(self, request, id):
        user = get_object_or_404(User, id=id)
        user.is_active = True
        user.extended_date = timezone.now()
        user.save()
        return Response({"message": "user activated"})


class ModelPermissions(APIView):
    permission_classes = [AdminIdentity]

    def put(self, request):
        email = request.data.get("email")
        user = User.objects.get(email=email)
        permissions = request.data.get("model_permissions")
        user.model_permissions = permissions
        user.save()
        return Response({"message": "permissions updated"})


class IsAdmin(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        emails = config("ADMIN_EMAILS")
        emails = emails.split(",")
        for email in emails:
            if request.user.email == email:
                user = User.objects.get(email=request.user.email)
                serializer = UserSerializer(user)
                user.is_admin = True
                user.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(
            {"error": "you dont have permissions to update admin"},
            status=status.HTTP_400_BAD_REQUEST,
        )


class UserDeleteByAdmin(APIView):
    permission_classes = [AdminIdentity]

    def get_object(self, pk):
        try:
            return User.objects.get(pk=pk)
        except User.DoesNotExist:
            raise Http404

    def delete(self, request, pk, format=None):
        """
        Delete user by admin.
        """
        user = self.get_object(pk)
        serializer = UserSerializer(user)
        user_data = serializer.data
        subscriptions = StripeSubcription.objects.filter(
            stripe_customer_id=user_data.stripe_customer_id
        )
        for subscription in subscriptions:
            subscription.delete()
        user.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)
