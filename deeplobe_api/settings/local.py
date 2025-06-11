"""Development settings and globals."""

from __future__ import absolute_import

from .common import *  # noqa
from decouple import config

DEBUG = True

# EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "deeplobe",
        "USER": "",
        "PASSWORD": "",
        "HOST": "",
    }
}


CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

INSTALLED_APPS += ("debug_toolbar",)

MIDDLEWARE += ("debug_toolbar.middleware.DebugToolbarMiddleware",)

DEBUG_TOOLBAR_PATCH_SETTINGS = False

INTERNAL_IPS = ("127.0.0.1",)

CORS_ORIGIN_ALLOW_ALL = True

REDIS_PORT = config("REDIS_PORT")
REDIS_HOST = config("REDIS_HOST")

RQ_QUEUES = {
    "default": {
        "HOST": REDIS_HOST,
        "PORT": REDIS_PORT,
        "DB": 0,
    }
}


# # The AWS region to connect to.
# AWS_REGION = config("AWS_REGION")
# # The AWS access key to use.
# AWS_ACCESS_KEY_ID = config("AWS_ACCESS_KEY_ID")
# # The AWS secret access key to use.
# AWS_SECRET_ACCESS_KEY = config("AWS_SECRET_ACCESS_KEY")
# # The name of the bucket to store files in.
# AWS_S3_BUCKET_NAME = config("AWS_S3_BUCKET_NAME")
# # To upload  media files to S3
# DEFAULT_FILE_STORAGE = "django_s3_storage.storage.S3Storage"
# AWS_S3_BUCKET_AUTH = False
# AWS_DEFAULT_ACL = "public-read"
