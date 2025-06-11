import os
import datetime
import requests
import filetype
import uuid
import threading

from io import BytesIO

from django.conf import settings

from deeplobe_api.utils.emails import emails

from deeplobe_ai.tagger import TaggerPredict
from deeplobe_ai.semantic import predict_img
from deeplobe_ai.instance import image_prediction
from deeplobe_ai.similarity import predict as similarity_predict
from deeplobe_ai.classification import predict as classification_predict
from deeplobe_ai.llm import Classifier
from deeplobe_api.db.models import AIModel, MediaAsset
from deeplobe_api.db.models import AIModelPrediction as AIModelPredictionModel
from deeplobe_api.api.views.function_calls.aimodels import SwitchModel

from deeplobe_api.api.views import save_frame_in_s3

BASE_DIR = settings.BASE_DIR


def video_prediction(aimodel, response, data, predict, save_predictions):
    uid = uuid.uuid4()
    with open(
        f"{save_predictions}/{uid}.mp4", "wb"
    ) as f:  # opening a file handler to create new file
        f.write(response.content)
    result = predict.predict_on_video(
        f"{save_predictions}/{uid}.mp4",
        f"{save_predictions}",
        uid,
        aimodel.uuid,
    )
    PredicitonResultUpdate(data.get("id"), result)
    AIModelPredictionMail(aimodel, data.get("request"), "video")


# @sync_to_async
def AIModelPrediction(data, aimodel):
    typ = "image"
    # shutil.rmtree(f"data/{aimodel.uuid}/prediction", ignore_errors=True)
    weight_file = f"{BASE_DIR}/{aimodel.weight_file}"
    save_predictions = f"{BASE_DIR}/data/{aimodel.uuid}/prediction"
    save_predictions_uuid = f"{BASE_DIR}/data/{aimodel.uuid}"
    os.makedirs(save_predictions, exist_ok=True)

    if aimodel.model_type == "object_detection":
        response = requests.get(data.get("input_images"))
        input_file = BytesIO(response.content)
        fileinfo = filetype.guess(input_file)  # the argument can be buffer or file
        content_type = fileinfo.mime
        extension = fileinfo.extension

        predict = TaggerPredict(
            weight_file,
            f"{aimodel.annotation_file.url}",
        )

        typ = content_type.split("/")[0]

        if typ == "image":
            result = predict.predict_on_image(
                input_file, save_predictions, aimodel.uuid
            )
        elif typ == "video":
            prediction = threading.Thread(
                target=video_prediction,
                args=[aimodel, response, data, predict, save_predictions],
            )
            prediction.start()
            result = None
        else:
            print("Invalid file: Input file is not an image nor a video.")
            return None

    if aimodel.model_type == "segmentation":
        coordinates = predict_img(
            data.get("input_images"),
            weight_file,
            aimodel.annotation_file.url,
            f"{save_predictions}/",
            f"{data.get('id')}.png",
        )
        body = open(f"{save_predictions}/{data.get('id')}.png", "rb")
        result = {
            "image": save_frame_in_s3(
                f"segmentation/{aimodel.uuid}/predictions/{data.get('id')}.png",
                body,
                "image/png",
            ),
            "json": coordinates["properties"],
        }

    if aimodel.model_type == "classification":
        classification_media = MediaAsset.objects.filter(name=aimodel.uuid)
        result = classification_predict(
            weight_file,
            classification_media,
            data.get("input_images"),
        )
        result = {"result": result}

    if aimodel.model_type == "image_similarity":
        result = similarity_predict(
            weight_file,
            # data.get("input_images")[0],
            data.get("input_images"),
            thresh=0.7,
        )
        result = {"result": result}

    if aimodel.model_type == "instance":
        coordinates = image_prediction(
            data.get("input_images"),
            weight_file,
            aimodel.annotation_file.url,
            f"{save_predictions}/",
            f"{data.get('id')}.png",
            0.7,
        )  # threshold ranges from 0 to 1, default = 0.1 if not mentioned
        body = open(f"{save_predictions}/{data.get('id')}.png", "rb")
        result = {
            "image": save_frame_in_s3(
                f"instance/{aimodel.uuid}/predictions/{data.get('id')}.png",
                body,
                "image/png",
            ),
            "json": coordinates["properties"],
        }
    if aimodel.model_type == "llm":
        llm_model = Classifier(save_predictions_uuid)
        result =llm_model.ask(
            data.get("input_images")
         
        )
        result = {"result": result}

    aimodel.last_used = datetime.datetime.now()
    aimodel.is_active = True
    aimodel.save()
    return PredicitonResultUpdate(data.get("id"), result)

    # AIModelPredictionMail(uuid, request)


def PredicitonResultUpdate(id, result):
    aimodel_pred_obj = AIModelPredictionModel.objects.get(id=id)
    aimodel_pred_obj.result = result
    aimodel_pred_obj.save()
    return aimodel_pred_obj


def AIModelPredictionMail(aimodel, request, typ):
    model_class = SwitchModel(aimodel.model_type)
    if aimodel.model_type == "object_detection":
        model_type = "object-detection"
        result = AIModelPredictionModel.objects.latest("id").result
        s3_url = result
        if typ == "video":
            emails.custom_model_prediction_email(
                request, aimodel, model_class, model_type, s3_url
            )
