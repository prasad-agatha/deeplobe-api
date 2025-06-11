from django.urls import path
from deeplobe_api.api.views.ocr_text_detection import OCRTextDetection

urlpatterns = [
    path("ocr_text_detection/", OCRTextDetection.as_view(), name="ocrtextdetection"),
]
