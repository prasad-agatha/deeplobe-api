from django.urls import path

from deeplobe_api.api.views.aimodel import (
    AllModels,
    AimodelImages,
    OCRSaveAIModel,
    OCRTrainingView,
    OCRPredictionView,
    AIModelDetailView,
    OCRAIModelDetails,
    AIModelListCreate,
    AIModelReTraining,
    AnnotationView,
    # DeleteAnnotationView,
    CustomModelDetailsAPI,
    AIModelPredictionView,
    CustomModelCreationAPI,
    CustomModelsTrainingAPI,
    AIModelPredictionDetail,
    AIModelOCRPredictionDetails,
    AIModelAnnotationReTraining,
    SaveAIModelOCRPredictionData,
 
)


urlpatterns = [
    # aimodels endpoints
    path("allmodels/", AllModels.as_view(), name="allmodels"),
    path("aimodels/", AIModelListCreate.as_view(), name="aimodels"),
    path("aimodels/<int:pk>", AIModelDetailView.as_view(), name="aimodel-detail"),
    path("ai-models/<str:uuid>", AimodelImages.as_view(), name="aimodel-images"),
    path("custom-models/", CustomModelCreationAPI.as_view(), name="create-custom"),
    path(
        "custom-models/<str:pk>/",
        CustomModelDetailsAPI.as_view(),
        name="custom-details",
    ),
    path(
        "custom-models-training/<str:pk>/",
        CustomModelsTrainingAPI.as_view(),
        name="custom-details",
    ),
    path(
        "aimodels/<int:pk>/prediction",
        AIModelPredictionView.as_view(),
        name="aimodels-prediction",
    ),
    path(
        "aimodels-prediction/<int:pk>/",
        AIModelPredictionDetail.as_view(),
        name="aimodels-prediction-detail",
    ),
    path(
        "ocr/save/aimodel/",
        OCRSaveAIModel.as_view(),
        name="ocr-save-aimodel",
    ),
    path(
        "ocr/aimodel/<int:pk>",
        OCRAIModelDetails.as_view(),
        name="ocr-aimodel-details",
    ),
    path(
        "ocr/aimodel/<int:pk>/prediction",
        SaveAIModelOCRPredictionData.as_view(),
        name="ocr-aimodel-details",
    ),
    path(
        "ocr/aimodel/prediction/<int:pk>",
        AIModelOCRPredictionDetails.as_view(),
        name="ocr-aimodel-prediction-details",
    ),
    path(
        "aimodels/<int:pk>/retraining/",
        AIModelReTraining.as_view(),
        name="model-retraining",
    ),
    path(
        "ocr/training/aimodels/",
        OCRTrainingView.as_view(),
        name="ocr-model-training",
    ),
    path(
        "ocr/prediction/aimodels/<int:pk>/",
        OCRPredictionView.as_view(),
        name="ocr-model-prediction",
    ),
    path(
        "aimodels/<int:pk>/retraining/annotation/",
        AIModelAnnotationReTraining.as_view(),
        name="model-annotation-retraining",
    ),
    path(
        "annotation-details/<str:pk>/<str:type>",
        AnnotationView.as_view(),
        name="annotation-detail",
    ),
    # path(
    #     "annotation-delete-details/<str:pk>/<str:type>",
    #     DeleteAnnotationView.as_view(),
    #     name="annotation-detail",
    # ),
]
