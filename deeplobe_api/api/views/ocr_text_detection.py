from django.conf import settings


from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
import os
import uuid

import cv2
import requests
import numpy as np
import warnings

from PIL import Image
from paddleocr import PaddleOCR

warnings.filterwarnings("ignore")


class Paddle_OCR:
    def __init__(self):
        self.det_path = "deeplobe_ai/models/paddle_ocr/ch_PP-OCRv3_det_infer"
        self.rec_path = "deeplobe_ai/models/paddle_ocr/ch_PP-OCRv3_rec_infer"
        self.cls_path = "deeplobe_ai/models/paddle_ocr/ch_ppocr_mobile_v2.0_cls_infer"

    def detect_text(self, image_url, coords):
        save_image_path = f"{settings.BASE_DIR}/{settings.MEDIA_ROOT}/paddleocr"
        os.makedirs(save_image_path, exist_ok=True)
        file_name = os.path.join(save_image_path, f"{uuid.uuid4()}.jpg")
        # Read the image and crop it
        image = Image.open(requests.get(image_url, stream=True).raw)
        image = np.array(image, dtype=int)

        # Split the coordinates string into a list of individual strings
        coords_list = coords.split(",")
        # Convert each element to an integer
        x, y, w, h = map(int, coords_list)

        # Cropping Image based on bbox
        image = image[y : y + h + 1, x : x + w + 1]
        cv2.imwrite(file_name, image)

        ocr = PaddleOCR(
            use_angle_cls=True,
            det_model_dir=self.det_path,
            rec_model_dir=self.rec_path,
            cls_model_dir=self.cls_path,
        )

        text = " ".join([line[1][0] for line in ocr.ocr(file_name)[0]])
        os.remove(file_name)
        return text


class OCRTextDetection(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        image_url = request.data.get("image_url")
        coords = request.data.get("coords")
        ocr = Paddle_OCR()
        try:
            result = ocr.detect_text(image_url, coords)
            return Response({"text": result}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
