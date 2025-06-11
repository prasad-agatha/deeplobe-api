from decouple import config

from django.utils import timezone

from rest_framework import permissions, status
from rest_framework.permissions import IsAdminUser
from rest_framework.exceptions import APIException, PermissionDenied

from deeplobe_api.db.models import User, APIKey, AIModel

from deeplobe_api.api.views.function_calls.aimodels import get_user_role


def annotator_permissions(self, request):
    user_role_permissions = get_user_role(request)
    if user_role_permissions["role"] == "annotator":
        raise PermissionDenied("You don't have permission to access this resource.")


class DomainNotValidException(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_detail = "Your domain must ends with @intellectdata.com or @soulpageit.com"


class DomainValidationPermissions(permissions.BasePermission):
    def has_permission(self, request, view):
        email = request.user.email
        if email.endswith("@intellectdata.com") or email.endswith("@soulpageit.com"):
            return True
        raise DomainNotValidException


class AdminIdentity(IsAdminUser):
    def has_permission(self, request, view):
        emails = config("ADMIN_EMAILS")
        emails = emails.split(",")
        for email in emails:
            if request.user.email == email:
                return True


class APIKeyExpiredException(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_detail = "Your API-Key has been expired."


class InvalidAPIKeyException(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_detail = "Your APIkey is invalid."


class APIKeyValidPermissions(permissions.BasePermission):
    def has_permission(self, request, view):
        api_key = request.headers.get("APIKEY")
        if api_key is None:
            return False
        try:
            apikey = APIKey.objects.get(key=api_key)
        except APIKey.DoesNotExist:
            raise InvalidAPIKeyException

        pk = view.kwargs.get("pk")
        ai_model = AIModel.objects.filter(id=pk).first()
        if ai_model is not None and ai_model.api_key == api_key:
            request.user = apikey.user
            return True
        else:
            request.user = None
            return False  # or raise an appropriate exception


        # if apikey and apikey.active == True:
        #     if timezone.now().date() > apikey.expire_date:
        #         apikey.active = False
        #         apikey.save()
        #         raise APIKeyExpiredException
        #     request.user = apikey.user
        #     return True
        # else:
        #     request.user = None
        #     raise APIKeyExpiredException
