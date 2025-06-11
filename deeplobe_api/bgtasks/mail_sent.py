import asyncio
from django.conf import settings
from asgiref.sync import sync_to_async
from django.core.mail import EmailMessage


async def mail_task(email, subject, body):

    loop = asyncio.get_event_loop()

    loop.create_task(mail_sending(email, subject, body))


@sync_to_async
def mail_sending(to_email, subject, body):
    email = EmailMessage(subject, body, settings.DEFAULT_FROM_EMAIL_SERVER, [to_email])
    email.content_subtype = "html"
    email.send()
