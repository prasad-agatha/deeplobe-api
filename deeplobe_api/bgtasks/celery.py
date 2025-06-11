from celery import Celery
from celery.schedules import crontab
import os
from decouple import config

# set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deeplobe_api.settings")
CELERY_BROKER_URL = config("CELERY_BROKER_URL")
app = Celery("deeplobe_api", broker=CELERY_BROKER_URL)
# broker="amqp://admin:mypass@rabbitmq:5672//
app.config_from_object("django.conf:settings", namespace="CELERY")
# Load task modules from all registered Django app configs.
app.autodiscover_tasks()
app.conf.timezone = "UTC"


app.conf.beat_schedule = {
    "run-every-day": {
        "task": "deeplobe_api.bgtasks.tasks.Last_7_days_inactive_models",
        "schedule": crontab(minute=0, hour=0),
    },
}

# app.conf.beat_schedule = {
#     "run-inactive-users-everyday": {
#         "task": "deeplobe_api.bgtasks.tasks.inactivate_users_after_15_days",
#         # run everyday at midnight
#         "schedule": crontab(minute=0, hour=0),
#     },
# }
