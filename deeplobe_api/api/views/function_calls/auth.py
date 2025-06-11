import requests

from rest_framework_jwt.settings import api_settings


jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER


def payload_generate_token(self, user):

    payload = jwt_payload_handler(user)

    token = jwt_encode_handler(payload)

    return token


def payload_generate(self, user):

    payload = jwt_payload_handler(user)

    token = jwt_encode_handler(payload)

    # user_data being passed into Response
    data = self.user_data(user, token)

    return data


def google_verify(request):

    try:
        # token = "ya29.a0AfH6SMDa54m4QZ--pwi7YfTqmm6HA-WNCj08ZUQMABQHfc6TGAxVvLNQuqtVYIbrMn-6bH68MZZHH4MHBG3GclzXH3I6cbHYH4WwQGBt4ZkqOUVu1Di3N8qw6pyfJNYjQpUdWG8P2F-sV6f5dOo0Q2mq-PikFxc"
        response = requests.get(
            f"https://www.googleapis.com/oauth2/v3/userinfo?access_token={request.data['token']}"
        )
        data = response.json()
        return data

    except Exception as e:
        print(e)
        return {"error": "invalid_request", "error_description": "Invalid Credentials"}
