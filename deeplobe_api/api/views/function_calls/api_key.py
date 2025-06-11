import secrets

from django.conf import settings

from rest_framework import status
from rest_framework.response import Response


from deeplobe_api.db.models import APIKey, AIModel
from deeplobe_api.utils.emails import emails
from deeplobe_api.api.serializers import APIKeySerializer



def create_api_key(request, api_key_limit):
    pretrained_models = {
        "pre_pose_detection",
        "pre_pii_extraction",
        "pre_text_moderation",
        "pre_facial_detection",
        "pre_table_extraction",
        "pre_image_similarity",
        "pre_wound_detection",
        "pre_facial_expression",
        "pre_sentimental_analysis",
        "pre_demographic_recognition",
        "pre_people_vehicle_detection",
        "pre_background_removal",
    }

    model_name = request.data.get("model_name", None)
    model_id = request.data.get("model_id", None)

    if model_name and model_name not in pretrained_models:
        return Response(
            {"msg": f"Invalid model name. Valid models are: {pretrained_models}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    token = secrets.token_hex(16) + secrets.token_hex(16)
    secret = settings.SECRET_KEY
    request.data["user"] = request.user.id
    request.data["key"] = token
    request.data["secret"] = secret

    if model_name:
        request.data["pretrained_model"] = model_name
        model_id = None
        request.data["aimodel"] = None  # Set aimodel field to None

    elif model_id:
        model = AIModel.objects.filter(id=model_id).first()
        if model is not None:
            request.data["pretrained_model"] = None  # Set pretrained_model field to None
            request.data["aimodel"] = model_id

            # Update the apikey field in the AIModel table
            new_apikey = token
            model.api_key = new_apikey
            model.save()

        else:
            return Response(
                {"msg": "Invalid model_id. Model not found."},
                status=status.HTTP_400_BAD_REQUEST,
            )

    else:
        return Response(
            {"msg": "Invalid payload. Either 'model_name' or 'model_id' must be provided."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    serializer = APIKeySerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()
        emails.api_key_generation_email(request)
        return Response(serializer.data)

    api_key_count = APIKey.objects.filter(user=request.user.id).count()
    if api_key_count >= api_key_limit:
        return Response(
            {"msg": f"User can't create more than {api_key_limit} API keys"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    return Response(
        {"msg": "Invalid payload. Model not found."},
        status=status.HTTP_400_BAD_REQUEST,
    )


