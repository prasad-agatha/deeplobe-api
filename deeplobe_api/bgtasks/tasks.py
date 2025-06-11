from deeplobe_api.bgtasks.celery import app
from celery.schedules import crontab
from deeplobe_api.db.models.aimodel import AIModel
from django.utils import timezone
from datetime import timedelta
from django.db.models import Q


@app.task
def add(x, y):
    z = x + y
    print(z)


@app.task
def Last_7_days_inactive_models():
    AIModel.objects.filter(
        Q(
            is_active=True,
            last_used__date__lt=(timezone.now() - timedelta(days=7)).date(),
        )
        | Q(
            is_active=True,
            last_used__isnull=True,
            created__date__lt=(timezone.now() - timedelta(days=7)).date(),
        )
    ).update(is_active=False)
    return "active flag updated"


# @app.task
# def inactivate_users_after_15_days():
#     from django.utils import timezone
#     from datetime import timedelta
#     from deeplobe_api.db.models import User

#     users = (
#         User.objects.filter(is_active=True)
#         .exclude(email__endswith="@intellectdata.com")
#         .exclude(email__endswith="@soulpageit.com")
#     )
#     for user in users:
#         try:
#             # if extended date is None take user joined date
#             if user.created and user.extended_date is None:
#                 last_date = (user.created + timedelta(days=15)).date()
#                 if timezone.now().date() > last_date:
#                     user.is_active = False
#                     user.save()
#             else:
#                 last_date = (user.extended_date + timedelta(days=15)).date()
#                 if timezone.now().date() > last_date:
#                     user.is_active = False
#                     user.save()
#         except Exception as e:
#             print(e)
