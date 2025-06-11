from .aws import save_frame_in_s3

from .state import StateList, StateDetail

from .cleaner import DataFolderCleanerEndpoint


from .country import CountryList, CountryDetail

from .filters import CountryFilter, StateFilter

from .ocr_text_detection import OCRTextDetection

from .register import RegisterList, RegisterDetail

from .api_count import APICountList, APICountDetail

from .function_calls.training import AIModelTraining

from .api_key import APIKeyEndpoint, UserAPIKeysDetail

from .function_calls.prediction import AIModelPrediction

from .contact_us import ContactUsViewSet, ContactUsDetails

from .my_models import ModelDownload, MyModels, ModelDetail

from .support_apis import UuidStatus, AllCategories, CheckName

from .stripe import (
    StripeSubcriptionDetail,
    StripeInvoiceList,
    StripeCreateSession,
    StripeCreatePortalSession,
)

from .auto_annotate_model import AutoAnnotateListView, AutoAnnotateDetailView

from .frames import (
    FramesAPIView,
    FramesCountAPIView,
    FramesToVideoAPIView,
    UpdateAnnotionsWithS3URLsAPIView,
)

from .annotation_hire_request import (
    AnnotationHireExpertRequestView,
    AnnotationHireExpertRequestDetails,
)

from .permissions import (
    AdminIdentity,
    DomainNotValidException,
    DomainValidationPermissions,
)


from .support import (
    SupportView,
    SupportDetails,
    SupportTicketView,
    SupportTicketListView,
)
from .media_asset import (
    ImageResize,
    MediaAssetView,
    ImageResizeEndpoint,
    MediaAssetDetailView,
)

from .pretrained_models import (
    PretrainedAPI,
    APILoggerView,
    APIStatistics,
    APILoggerViewAll,
    APIKeyModelsView,
    APIMetricsLogsView,
    APILoggerCustomDetails,
    APIStatisticsByTimePeriodLineChart,
    APIStatisticsByTimePeriodCircularChart,
)

from .subscription import (
    CollaboratorInvitation,
    SubscriberCollaboratorView,
    SubscriberCollaboratorDetail,
    WorkspaceCollaboratorsDeletion,
)
from .workspace import (
    WorkspaceView,
    WorkspaceDetail,
    WorkspaceUsersView,
    PersonalWorkspaceView,
)

from .user import (
    UserCreate,
    UserDetail,
    TaskResults,
    image_output,
    AllUsersListView,
    UserProfilePicUpdate,
)

from .chargebee import (
    ChargeBeeSession,
    ChargeBeeBillingInfo,
    ChargeBeeInvoiceList,
    ChargeBeeSubscriptionUpdate,
    ChargeBeeInvoicePDFDownload,
    ChargeBeeSubscriptionCancellation,
)

from .auth import (
    IsAdmin,
    Sendmagiclink,
    SocialAuthView,
    VerifyMagicLink,
    ModelPermissions,
    UserDeleteByAdmin,
    MagicLinkRegister,
    ActivateUserEndpoint,
    # UpdateFirstTimeUser,
)

from .aimodel import (
    AimodelImages,
    OCRSaveAIModel,
    AnnotationView,
    OCRTrainingView,
    OCRPredictionView,
    AIModelListCreate,
    AIModelDetailView,
    OCRAIModelDetails,
    AIModelReTraining,
    # DeleteAnnotationView,
    AIModelPredictionView,
    CustomModelDetailsAPI,
    CustomModelCreationAPI,
    CustomModelsTrainingAPI,
    AIModelOCRPredictionDetails,
    SaveAIModelOCRPredictionData,
    AIModelAnnotationReTraining,
)
