import asyncio

from decouple import config

from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.core.mail import send_mail, EmailMultiAlternatives


from deeplobe_api.bgtasks import mail_task
from deeplobe_api.api.views.function_calls.image import convert_file_django_object


def send_mail_to_users(to_email, subject, username, body):
    """
    This will allows send emails to users
    """
    email_from = config("DEFAULT_FROM_EMAIL")
    recipient_list = [to_email]
    subject = subject
    body = body
    email = EmailMessage(subject, body, email_from, recipient_list)
    email.content_subtype = "html"
    email.send()


class Email:
    def __init__(self):
        self.client_name = config("CLIENT_NAME")
        self.server_name = config("SERVER_NAME")
        self.server_team = config("SERVER_TEAM")
        self.from_mail_server = config("DEFAULT_FROM_EMAIL_SERVER")
        self.from_mail = config("DEFAULT_FROM_EMAIL")
        self.client_logo = config("CLIENT_LOGO")
        self.address_line1 = config("ADDRESS_LINE1")
        self.address_line2 = config("ADDRESS_LINE2")
        self.twitter_site = config("TWITTER_SITE")
        self.twitter_logo = config("TWITTER_LOGO")
        self.facebook_logo = config("FACEBOOK_LOGO")
        self.facebook_site = config("FACEBOOK_SITE")
        self.linkedin_logo = config("LINKEDIN_LOGO")
        self.linkedin_site = config("LINKEDIN_SITE")
        self.website = config("WEBSITE")
        self.domain = config("DOMAIN")
        self.mail_id = config("MAIL_ID")
        self.domain = config("DOMAIN")

    def custom_model_failed_email(self, aimodel, data, model_class, model_type):
        model = aimodel.name
        email = data.get("email", None)
        to_mail = [email]
        template = "mail/model_fail_template.html"
        subject = f"{self.server_name} - Custom {model_class} model request: {model}"
        body = render_to_string(
            template,
            {
                "name": data.get("username"),
                "model": model,
                "domain": self.domain,
                "uuid": aimodel.id,
                "model_type": model_type,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(subject, body, self.from_mail_server, to_mail, html_message=body)

    def custom_model_trained_email(
        self,
        request,
        aimodel,
        model_class,
        model_type,
    ):
        model = aimodel.name
        email = request.get("email")
        subject = f"{self.server_name} - {model_class} model request:`{model}`"
        to_mail = [email]
        template = "mail/training_completion_page.html"
        body = render_to_string(
            template,
            {
                "name": request.get("username"),
                "model": model,
                "model_class": model_class,
                "domain": self.domain,
                "uuid": aimodel.uuid,
                "model_type": model_type,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(
            subject,
            body,
            self.from_mail_server,
            to_mail,
            html_message=body,
        )

    def custom_model_prediction_email(
        self,
        request,
        aimodel,
        model_class,
        model_type,
        s3_url,
    ):
        model = aimodel.name
        email = aimodel.user.email
        to_mail = [email]
        subject = f"{self.server_name} - {model_class} model request: {model}"
        template = "mail/prediction_completion_page.html"
        body = render_to_string(
            template,
            {
                "name": aimodel.user.username,
                "url": s3_url,
                "model": model,
                "model_class": model_class,
                "domain": self.domain,
                "uuid": aimodel.uuid,
                "model_type": model_type,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(
            subject,
            body,
            self.from_mail_server,
            to_mail,
            html_message=body,
        )

    def hire_annotation_expert_email(self, request, user):
        email = user.email
        to_mail = [email]
        template = "mail/hire_annotation_expert.html"
        subject = f"{self.server_name} - Annotation Request"
        body = render_to_string(
            template,
            {
                "user": request.user.username,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(
            subject,
            body,
            self.from_mail_server,
            to_mail,
            html_message=body,
        )

    def api_key_generation_email(self, request):
        email = request.user.email
        to_mail = [email]
        template = "mail/generate-api-page.html"
        subject = f"{self.client_name}: API is generated successfully"
        body = render_to_string(
            template,
            {
                "user": request.user.username,
                # "api": serializer.data["key"],
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "mail_id": self.mail_id,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def send_magic_link(self, email, user, token):
        subject = f"{self.server_name} - Sign in request link"
        to_mail = [email]
        template = "mail/sign-in-page.html"
        body = render_to_string(
            template,
            {
                "username": user.username,
                "token": token,
                "domain": self.domain,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(
            subject,
            body,
            self.from_mail,
            to_mail,
            html_message=body,
            fail_silently=False,
        )

    def magic_link_register(self, email, user, token):
        to_mail = [email]
        subject = f"{self.client_name}: Sign up request link"
        template = "mail/sign-up-page.html"
        body = render_to_string(
            template,
            {
                "username": user.username,
                "token": token,
                "domain": self.domain,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def model_deactivate_or_delete_email(self, request, weight_file):
        email = request.user.email
        to_mail = [email]
        template = "mail/deactive-delete-page.html"
        subject = f"{self.client_name}: Your model has been deleted"
        body = render_to_string(
            template,
            {
                "user": request.user.username,
                "model": weight_file.model_name,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "mail_id": self.mail_id,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def signup_new_users_email(self, collaborator_email):
        to_mail = [collaborator_email]
        subject = f"{self.client_name}: Sign up request link"
        template = "mail/signup_new_users.html"
        body = render_to_string(
            template,
            {
                "domain": self.domain,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def new_user_invitation_collaborator_email(
        self, user, invitee_subscription, invitee_workspace, new_user_invitation
    ):
        full_name = user.username
        subscription_id = invitee_subscription
        workspace_id = invitee_workspace.id
        workspace_name = invitee_workspace.name
        template = "mail/collaborator_invitation.html"
        subject = f"Invitation to Collaborate on {workspace_name}"
        body = render_to_string(
            template,
            {
                "domain": self.domain,
                "collaborator_name": full_name,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "subscription": subscription_id,
                "workspace": workspace_id,
                "workspace_name": workspace_name,
            },
        )
        send_mail(
            subject,
            body,
            self.from_mail_server,
            [new_user_invitation.collaborator_email],
            html_message=body,
        )

    def user_invitation_collaborator_email(
        self, request, subscription, workspace, role
    ):
        full_name = request.data.get("full_name")
        subscription_id = subscription
        workspace_id = workspace.id
        workspace_name = workspace.name
        subject = f"Invitation to {role} on {workspace_name}"
        template = "mail/collaborator_invitation.html"
        body = render_to_string(
            template,
            {
                "domain": self.domain,
                "role": role,
                "collaborator_name": full_name,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
                "subscription": subscription_id,
                "workspace": workspace_id,
                "workspace_name": workspace_name,
            },
        )

        send_mail(
            subject,
            body,
            self.from_mail,
            [request.data.get("email")],
            html_message=body,
        )

    def collaborator_invitation_rejected_email(self, collaborator, collaborator_name):
        full_name = collaborator_name
        to_mail = [collaborator.invitee]
        subject = "Collaborator invitation rejected"
        template = ("mail/collaborator_invitation_rejected.html",)
        body = render_to_string(
            template,
            {
                "domain": self.domain,
                "invitee": collaborator.invitee.username,
                "collaborator_name": full_name,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_name": self.client_name,
                "client_logo": self.client_logo,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facbook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )

        send_mail(
            subject,
            body,
            self.from_mail,
            to_mail,
            html_message=body,
        )

    def support_email_for_users(self, request, user, input_subject):
        email = user.email
        subject = f"{self.server_name} - Support request: {input_subject} "
        to_mail = [email]
        template = "mail/support.html"
        body = render_to_string(
            template,
            {
                "user": request.user.username,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(
            subject,
            body,
            self.from_mail_server,
            to_mail,
            html_message=body,
        )

    def support_email_for_team(self, request, input_subject):
        if self.client_name == "INFER":
            to_mail = ["support@intellectdata.com"]

        else:
            to_mail = ["contact@soulpageit.com"]

        ip = request.META.get("REMOTE_ADDR")
        subject = f"{self.server_name} - Support request: {input_subject}"
        template = ("mail/support_team.html",)
        body = render_to_string(
            template,
            {
                "user": request.user.username,
                "contact_number": request.user.contact_number,
                "email": request.user.email,
                "role": request.user.role,
                "company": request.user.company,
                "industry": request.user.industry,
                "subject": request.data.get("subject"),
                "description": request.data.get("description"),
                "file": request.data.get("file"),
                "ip": ip,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        try:
            mail = EmailMultiAlternatives(subject, body, self.from_mail_server, to_mail)

            file = request.data.get("file")
            if file:
                # f = convert_file_django_object(attachments)
                mail.attach(file.name, file.read(), file.content_type)
                mail.attach_alternative(body, "text/html")
                mail.send()
            else:
                mail.attach_alternative(body, "text/html")
                mail.send()
        except:
            pass

    def support_ticket_closed_email(self, request, user, ticket_id):
        email = user.email
        subject = "Support ticket closed"
        to_mail = [email]
        template = "mail/support_ticket_closed.html"
        body = render_to_string(
            template,
            {
                "user": request.user.username,
                "ticket_id": ticket_id,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)

    def contact_us_email_for_users(self, request, serializer_data, email):
        input_hearing = request.data.get("hearing")
        if serializer_data.title == "API implementation request":
            subject = (
                f"{self.server_name} - API implementation request: {input_hearing}"
            )

        else:
            subject = f"{self.server_name} - Custom model request:  {input_hearing}"
        template = "mail/contact_us.html"
        body = render_to_string(
            template,
            {
                "name": request.user.username,
                "title": serializer_data.title,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(mail_task(email, subject, body))

    def api_implementation_request_contact_us_team_email(
        self, request, serializer_data
    ):
        ip = request.META.get("REMOTE_ADDR")
        input_hearing = request.data.get("hearing")
        template = "mail/contact_us_team.html"
        subject = f"{self.server_name} - API implementation request: {input_hearing}"
        body = render_to_string(
            template,
            {
                "title": serializer_data.title,
                "name": request.user.username,
                "contact_number": request.user.contact_number,
                "email": request.user.email,
                "role": request.user.role,
                "company": request.user.company,
                "industry": request.user.industry,
                "subject": request.data.get("subject"),
                "server_name": self.server_name,
                "client_name": self.client_name,
                "server_team": self.server_team,
                "id": serializer_data.id,
                "description": serializer_data.description,
                "hearing": serializer_data.hearing,
                "created": serializer_data.created,
                "ip": ip,
                "client_logo": self.client_logo,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            mail_task(
                "contact@soulpageit.com"
                if self.client_name == "DeepLobe"
                else "support@intellectdata.com",
                subject,
                body,
            )
        )
        loop.close()

    def custom_model_request_contact_us_team_email(self, request, serializer_data):
        ip = request.META.get("REMOTE_ADDR")
        input_hearing = request.data.get("hearing")
        subject = f"{self.server_name} - Custom model request:  {input_hearing}"
        template = ("mail/contact_us_team.html",)
        body = render_to_string(
            template,
            {
                "title": serializer_data.title.lower(),
                "name": request.user.username,
                "contact_number": request.user.contact_number,
                "email": request.user.email,
                "role": request.user.role,
                "company": request.user.company,
                "industry": request.user.industry,
                "subject": request.data.get("subject"),
                "server_name": self.server_name,
                "client_name": self.client_name,
                "server_team": self.server_team,
                "id": serializer_data.id,
                "description": serializer_data.description,
                "hearing": serializer_data.hearing,
                "created": serializer_data.created,
                "ip": ip,
                "client_logo": self.client_logo,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            mail_task(
                "contact@soulpageit.com"
                if self.client_name == "DeepLobe"
                else "support@intellectdata.com",
                subject,
                body,
            )
        )

        loop.close()
    
       
    def stripe_subscription_created_email(self, request, user):
        email = user.email
        subject = "New Stripe Subscription Created"
        to_mail = [email]
        template = "mail/stripe_subscription_created.html"
        body = render_to_string(
            template,
            {
                "user": user.username,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)
    
    def stripe_subscription_plan_change_email(self, request, user,user_subscription):
        email = user.email
        subject = "Stripe Subscription Plan Changed"
        to_mail = [email]
        template = "mail/stripe_subscription_plan_change.html"
        body = render_to_string(
            template,
            {
                "user": user.username,
                "subscription_plan":user_subscription.plan_name,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)
        
    def stripe_subscription_plan_delete_email(self, request, user,user_subscription):
        email = user.email
        subject = "Stripe Subscription Plan Cancelled"
        to_mail = [email]
        template = "mail/stripe_subscription_plan_cancelled.html"
        body = render_to_string(
            template,
            {
                "user": user.username,
                "subscription_plan":user_subscription.plan_name,
                "server_name": self.server_name,
                "server_team": self.server_team,
                "client_logo": self.client_logo,
                "client_name": self.client_name,
                "address_line1": self.address_line1,
                "address_line2": self.address_line2,
                "twitter_site": self.twitter_site,
                "twitter_logo": self.twitter_logo,
                "facebook_site": self.facebook_site,
                "facebook_logo": self.facebook_logo,
                "linkedin_site": self.linkedin_site,
                "linkedin_logo": self.linkedin_logo,
                "website": self.website,
            },
        )
        send_mail(subject, body, self.from_mail, to_mail, html_message=body)




emails = Email()
