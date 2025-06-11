from django.contrib import admin

# Register your models here.
from .models.user import User

from .models.task import Task

from .models.state import State

from .models.api_key import APIKey

from .models.country import Country

from .models.api_count import APICount

from .models.workspace import Workspace

from .models.auth import SocialProvider

from .models.contact_us import ContactUs

from .models.statistics import Statistic

from .models.api_loggers import APILogger

from .models.file_upload import FileAssets

from .models.task_status import TaskStatus

from .models.media_asset import MediaAsset

from .models.stripe import StripeSubcription, SubcriptionInvoices

from .models.support import Support, SupportTicket

from .models.aimodel import AIModel, AIModelPrediction

from .models.annotation_hire_request import AnnotationHireExpertRequest

from .models.subscription import Subscription, UserInvitation


admin.site.register(User)
admin.site.register(Task)
admin.site.register(State)
admin.site.register(APIKey)
admin.site.register(AIModel)
admin.site.register(Country)
admin.site.register(Support)
admin.site.register(APICount)
admin.site.register(ContactUs)
admin.site.register(Statistic)
admin.site.register(APILogger)
admin.site.register(Workspace)
admin.site.register(TaskStatus)
admin.site.register(MediaAsset)
admin.site.register(FileAssets)
admin.site.register(Subscription)
admin.site.register(SupportTicket)
admin.site.register(SocialProvider)
admin.site.register(UserInvitation)
admin.site.register(StripeSubcription)
admin.site.register(SubcriptionInvoices)
admin.site.register(AIModelPrediction)
admin.site.register(AnnotationHireExpertRequest)
