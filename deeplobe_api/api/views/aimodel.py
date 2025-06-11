import os
import json
import uuid
import wget
import requests
import threading
import django_rq

from decouple import config

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.http import Http404
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.core.files.base import ContentFile
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.contrib.sites.models import Site

from deeplobe_api.db.models import (
    APIKey,
    AIModel,
    APICount,
    Workspace,
    APILogger,
    MediaAsset,
    AIModelPrediction,
)

from deeplobe_api.api.serializers import (
    AIModelSerializer,
    MediaAssetSerializer,
    CustomAllModelSerializer,
    AIModelPredictionSerializer,
    CustomModelCreationSerializer,
    CustomModelUpdationSerializer,
)


from deeplobe_api.api.views.function_calls.aimodels import (
    SwitchModel,
    throw_error,
    get_user_role,
    get_annotations,
    failure_call_back,
    post_annotations_data,
    update_annotation_file,
    update_annotations_data,
    delete_annotations_data,
    check_model_name_availability,
)
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails
from deeplobe_api.api.views.permissions import (
    APIKeyValidPermissions,
    annotator_permissions,
)

from deeplobe_api.api.views.function_calls.training import AIModelTraining
from deeplobe_api.api.views.function_calls.prediction import (
    AIModelPrediction as AIModelPredictionTask,
)


class AllModels(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = AIModel.objects.filter(workspace=request.user.current_workspace)
        personal_workspace_id = request.query_params.get("personal")
        if personal_workspace_id:
            personal_workspace = Workspace.objects.filter(user=request.user).first()
            queryset = AIModel.objects.filter(workspace=personal_workspace.id)
            if request.query_params.get("exclude"):
                queryset = queryset.exclude(
                    model_type__in=["image_similarity", "classification"]
                )
        serializer = CustomAllModelSerializer(queryset, many=True)
        return Response(serializer.data)


class CustomModelCreationAPI(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = AIModel.objects.filter(workspace=request.user.current_workspace)
        user_role_permissions = get_user_role(request)
        model_ids = user_role_permissions["models"]["select"]
        if user_role_permissions["role"] == "annotator":
            if len(model_ids) > 1 or (len(model_ids) == 1 and model_ids[0] != "All"):
                queryset = queryset.filter(id__in=model_ids)
            # Exclude models with model_type "image_similarity" and "classification"
            queryset = queryset.exclude(
                model_type__in=["image_similarity", "classification"]
            )
        serializer = CustomModelCreationSerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request):
        # annotator_permissions(self, request)
        workspace_id = request.user.current_workspace
        workspace_obj = Workspace.objects.filter(id=workspace_id).first()
        current_workspace_details = chargebeeplandetails(email=workspace_obj.user)
        plan = current_workspace_details.get("plan", None)
        if plan != config("STRIPE_FREE_PLAN"):
            # check name is available or not
            name = request.data.get("name")
            if name:
                err = check_model_name_availability(request, name)
                if err:
                    return Response({"msg": err}, status=status.HTTP_400_BAD_REQUEST)

            request.data["user"] = request.user.id
            request.data["workspace"] = request.user.current_workspace
            request.data["uuid"] = uuid.uuid4().hex
            serializer = CustomModelCreationSerializer(data=request.data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                return Response(serializer.data)
        else:
            return Response(
                {
                    "msg": " Free-plan  doesn't have custom model access"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )


class CustomModelDetailsAPI(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        queryset = AIModel.objects.filter(
            uuid=pk, workspace=request.user.current_workspace
        ).first()
        if queryset is not None:
            serializer = CustomModelCreationSerializer(queryset)
            data = dict(serializer.data)
            if data["annotation_file"]:
                annotation_data = requests.get(data["annotation_file"]).json()
                data["annotation_details"] = {
                    "categories": annotation_data["categories"]
                }
            elif data["model_type"] == "classification" or "image_similarity":
                queryset = MediaAsset.objects.filter(name=data["uuid"]).distinct(
                    "class_name"
                )
                media_serializer = MediaAssetSerializer(queryset, many=True)
                class_names = []
                total = 0
                for asset in media_serializer.data:
                    count = queryset = MediaAsset.objects.filter(
                        name=data["uuid"], class_name=asset["class_name"]
                    ).count()
                    class_names.append({"name": asset["class_name"], "count": count})
                    total = total + count
                data["annotation_details"] = {"categories": class_names, "total": total}

            else:
                data["annotation_details"] = {"categories": []}
            return Response(data, status=status.HTTP_200_OK)
        else:
            return Response(
                {"msg": "Model not found"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    queryset = AIModel.objects.all()
    serializer_class = CustomModelUpdationSerializer

    def put(self, request, pk):
        # get the AIModel with the given pk and current workspace
        queryset = self.get_object()
        if queryset.workspace != request.user.current_workspace:
            return Response(
                {"msg": "Not authorized."}, status=status.HTTP_401_UNAUTHORIZED
            )

        # check if model name already exists
        name = request.data.get("name")
        if name:
            err = check_model_name_availability(request, name)
            if err:
                return Response({"msg": err}, status=status.HTTP_400_BAD_REQUEST)

        if queryset.annotation_file:
            try:
                # update annotation file if it exists in payload
                annotation_file = request.data.get("annotation_file")
                if annotation_file:
                    update_annotation_file(
                        request=request, model=queryset, annotation_file=annotation_file
                    )
                    return Response({"msg": "Annotation file updated successfully."})
            except Exception as e:
                return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # update model if no annotation file to update
        serializer = self.get_serializer(queryset, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        queryset = AIModel.objects.filter(
            uuid=pk, workspace=request.user.current_workspace
        ).first()
        queryset.status = "Deleted"
        queryset.save()
        return Response({"msg": "model deleted"})


class CustomModelsTrainingAPI(APIView):
    def get(self, request, pk):
        ai_model = AIModel.objects.filter(
            uuid=pk, workspace=request.user.current_workspace
        ).first()
        if ai_model is not None:
            serializer = AIModelSerializer(ai_model)
            data = dict(serializer.data)

            return Response(data)
        else:
            return Response(
                {"msg": "Model not found"}, status=status.HTTP_400_BAD_REQUEST
            )

    def post(self, request, pk):
        
        user_data = {
            "origin": request.headers.get("Origin"),
            "email": request.user.email,
            "username": request.user.username,
            "training_mode": request.data.get("training_mode", "production"),
          
        }

        ai_model = AIModel.objects.filter(uuid=pk, workspace=request.user.current_workspace).first()
        workspace_id = request.user.current_workspace
        workspace_obj = Workspace.objects.filter(id=workspace_id).first()
        if ai_model is not None:
            if chargebeeplandetails(email=workspace_obj.user).get("plan") != config("STRIPE_FREE_PLAN")  :
                # Check the number of existing models for the user
                user_models_count = AIModel.objects.filter(workspace=request.user.current_workspace).count()

                if user_models_count <= 6:
                    os.makedirs(f"{settings.BASE_DIR}/data/{pk}", exist_ok=True)

                    serializer = AIModelSerializer(ai_model)
                    data = dict(serializer.data)

                    if (
                        data["model_type"] == "classification"
                        or data["model_type"] == "image_similarity"
                        or data["model_type"] == "llm"
                    ):
                        ai_model.is_trained = False
                        ai_model.save()
                        django_rq.enqueue(
                            AIModelTraining,
                            pk,
                            user_data,
                            job_timeout=18000,
                            on_failure=failure_call_back,
                        )      
                        return Response(data)

                    annotation_file = requests.get(data["annotation_file"]).json()
                    annotations = list(annotation_file["annotations"])
                    img_unann = [
                        x for x in list(annotation_file["images"]) if (x["annotated"] is False)
                    ]

                    if len(img_unann) > 0:
                        return throw_error("Annotate all images to train model")

                    categories = list(annotation_file["categories"])
                    categories = [
                        x for x in categories if (str("deleted_") not in x.get("name"))
                    ]
                    updated_categories = []

                    for index, cat in enumerate(categories):
                        updated_categories.append(
                            {"previous_id": cat["id"], "updated_id": index}
                        )
                        cat["id"] = index

                    for index, ann in enumerate(annotations):
                        ann["id"] = index + 1
                        ann["category_id"] = [
                            x.get("updated_id")
                            for x in updated_categories
                            if (ann["category_id"] == x.get("previous_id"))
                        ][0]

                    annotations_data = {
                        "categories": categories,
                        "annotations": annotations,
                        "images": annotation_file["images"],
                    }

                    ai_model.annotation_file.save(
                        "Annotate.json",
                        ContentFile(json.dumps(annotations_data, indent=2).encode("utf-8")),
                        save=True,
                    )

                    ai_model.is_trained = False
                    ai_model.save()

                    if ai_model.model_type == "ocr":
                        domain = Site.objects.get_current().domain
                        url = f"https://{domain}"
                        ocr_user_data = {
                            "origin": url,
                            "email": ai_model.user.email,
                            "username": ai_model.user.username,
                            "model_type": ai_model.model_type,
                            "annotation_file": ai_model.annotation_file.url,
                            "aimodel_id": ai_model.id,
                            "aimodel_name": ai_model.name,
                            "uuid": ai_model.uuid,
                        }
                        base_url = config("MICRO_SERVICE_HOST")
                        endpoint = f"{base_url}/api/aimodels/"

                        try:
                            response = requests.post(url=endpoint, json=ocr_user_data)
                            aimodel_ocr = response.json()
                            return Response(aimodel_ocr)
                        except Exception as e:
                            print(str(e))
                    else:
                        django_rq.enqueue(
                            AIModelTraining,
                            pk,
                            user_data,
                            job_timeout=18000,
                            on_failure=failure_call_back,
                        )
                    return Response(data)
                else:
                    return Response(
                        {"msg": "Maximum models limit reached. Upgrade your plan."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            else:
                return Response(
                    {"msg": "upgrade to growth plan."},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            return Response(
                {"msg": "Model not found"},
                status=status.HTTP_400_BAD_REQUEST
            )

class AIModelListCreate(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        aimodel = AIModel.objects.filter(workspace=request.user.current_workspace)
        serializer = AIModelSerializer(aimodel, many=True)
        return Response(serializer.data)

    def post(self, request):
        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()
        meta_data = chargebeeplandetails(email=workspace.user)
        api_count = APICount.objects.filter(workspace=workspace).exists()

        dict_data = {
            "uuid": request.data.get("uuid"),
            "user": request.user.id,
            "annotation_file": request.data.get("annotation_file"),
            "name": request.data.get("name"),
            "description": request.data.get("description"),
            "model_type": request.data.get("model_type"),
            "workspace": workspace.id,
        }
        serializer = AIModelSerializer(data=dict_data)
        if serializer.is_valid():
            serializer.save()
            user_data = {
                "origin": request.headers.get("Origin"),
                "email": request.user.email,
                "username": request.user.username,
            }

            job = django_rq.enqueue(
                AIModelTraining,
                request.data.get("uuid"),
                user_data,
                job_timeout=18000,
                on_failure=failure_call_back,
            )

            serializer.save(job={"id": job._id, "status": job._status})

            if api_count:
                custom_model_count = APICount.objects.get(
                    workspace=workspace.id,
                    created__date__range=[
                        meta_data["from_date"],
                        meta_data["to_date"],
                    ],
                )
                if not (
                    meta_data["api-requests"]
                    >= custom_model_count.custom_model_api_count
                    and meta_data["plan_status"] == "active"
                ):
                    return Response("plan exceeded")
                custom_model_count.custom_model_api_count = (
                    custom_model_count.custom_model_api_count + 1
                )
                custom_model_count.save()

                APICount.objects.create(workspace=workspace, custom_model_api_count=1)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class OCRSaveAIModel(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()
        meta_data = chargebeeplandetails(email=workspace.user)
        api_count = APICount.objects.filter(workspace=workspace).first()

        dict_data = {
            "uuid": request.data.get("uuid"),
            "user": request.user.id,
            "annotation_file": request.data.get("annotation_file"),
            "name": request.data.get("name"),
            "description": request.data.get("description"),
            "model_type": request.data.get("model_type"),
            "workspace": workspace.id,
        }
        serializer = AIModelSerializer(data=dict_data)
        if serializer.is_valid():
            serializer.save()
            if api_count:
                custom_model_count = api_count.custom_model_api_count
                if not (
                    meta_data["api-requests"] >= custom_model_count
                    and meta_data["plan_status"] == "active"
                ):
                    return Response("plan exceeded")
                api_count.custom_model_api_count = custom_model_count + 1
                api_count.save()

            else:
                APICount.objects.create(workspace=workspace, custom_model_api_count=1)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class OCRAIModelDetails(APIView):
    def get_object(self, pk):
        try:
            return AIModel.objects.get(pk=pk)
        except AIModel.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        aimodel = self.get_object(pk)
        serializer = AIModelSerializer(aimodel)
        return Response(serializer.data)

    def put(self, request, pk):
        aimodel = self.get_object(pk)
        serializer = AIModelSerializer(aimodel, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        aimodel = self.get_object(pk)
        aimodel.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class AIModelDetailView(APIView):
    permission_classes = (IsAuthenticated,)

    """
    Retrieve or delete a AIModels team instance.
    """

    def get_object(self, pk):
        try:
            return AIModel.objects.get(pk=pk)
        except AIModel.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        aimodel = AIModel.objects.filter(
            workspace=request.user.current_workspace, id=pk
        ).first()
        if aimodel is None:
            return Response("Model not found", status=status.HTTP_400_BAD_REQUEST)
        serializer = AIModelSerializer(aimodel)
        data = dict(serializer.data)
        if data["annotation_file"]:
            annotation_data = requests.get(data["annotation_file"]).json()
            data["annotation_details"] = {
                "categories": annotation_data["categories"],
                "images": len(annotation_data["images"]),
            }
        elif data["model_type"] == "classification" or "image_similarity":
            queryset = MediaAsset.objects.filter(name=data["uuid"]).distinct(
                "class_name"
            )
            media_serializer = MediaAssetSerializer(queryset, many=True)
            class_names = []
            for asset in media_serializer.data:
                class_names.append(asset["class_name"])
            data["annotation_details"] = {"categories": class_names}
        else:
            data["annotation_details"] = {"categories": [], "images": 0}
        return Response(data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        aimodel = self.get_object(pk)
        serializer = AIModelSerializer(aimodel, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        aimodel = self.get_object(pk)
        aimodel.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class AimodelImages(APIView):

    def post(self, request, uuid):
        class_name = request.data.get("class_name")
        assets = request.data.getlist("asset")  # Get the list of image files

        data_to_save = []
        for asset in assets:
            dict_data = {"name": uuid, "asset": asset, "class_name": class_name}
            serializer = MediaAssetSerializer(data=dict_data)
            if serializer.is_valid():
                serializer.save()
                data_to_save.append(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        return Response(data_to_save, status=status.HTTP_200_OK)

class AIModelPredictionDetail(APIView):
    permission_classes = [APIKeyValidPermissions | IsAuthenticated]

    def get(self, request, pk):
        queryset = get_object_or_404(AIModelPrediction, id=pk)
        serializer = AIModelPredictionSerializer(queryset)
        return Response(serializer.data, status=status.HTTP_200_OK)


class AIModelPredictionView(APIView):
    permission_classes = [APIKeyValidPermissions | IsAuthenticated]

    def ckpt_download_thread(self, aimodel_, local_path):
        wget.download(aimodel_, out=local_path)

    def get_object(self, pk):
        try:
            return AIModelPrediction.objects.get(pk=pk)
        except AIModelPrediction.DoesNotExist:
            raise Http404

    def check_weight_file(self, aimodel):
        BASE_DIR = settings.BASE_DIR
        if not os.path.exists(f"{BASE_DIR}/{aimodel.weight_file}"):
            print("not Exist in local")
            local_path = f"{BASE_DIR}/data/{aimodel.uuid}"
            os.makedirs(local_path, exist_ok=True)
            download = threading.Thread(
                target=self.ckpt_download_thread,
                args=[aimodel.aws_weight_file, local_path],
            )
            download.start()
            return False
        print("Ckpt file existed in local")
        return True

    def post(self, request, pk):
        aimodel = get_object_or_404(AIModel, id=pk)
        if not aimodel.model_type == "ocr" and not aimodel.model_type == "llm":
            is_exist = self.check_weight_file(aimodel)
            if not is_exist:
                return Response(
                    "Ckpt file is downloading please get back after few minutes",
                    status=status.HTTP_400_BAD_REQUEST,
                )
        input_url = request.data.get("input_url", None)
        dict_data = {
            "aimodel": pk,
            "input_images": input_url,
            "user": request.user.id,
            "description": request.data.get("description"),
        }

        serializer = AIModelPredictionSerializer(data=dict_data)
        if serializer.is_valid():
            serializer.save()
            if aimodel.model_type == "ocr":
                domain = Site.objects.get_current().domain
                url = f"https://{domain}"
                base_url = config("MICRO_SERVICE_HOST")
                endpoint = f"{base_url}/api/aimodels/{pk}/prediction"

                try:
                    response = requests.post(
                        url=endpoint,
                        json={
                            "origin": url,
                            "input_image": input_url,
                            "aimodel_prediction_id": serializer.data.get("id"),
                            "aimodel": {
                                "id": aimodel.id,
                                "uuid": aimodel.uuid,
                                "weight_file": aimodel.weight_file,
                                "aws_weight_file": aimodel.aws_weight_file,
                                "annotation_file": aimodel.annotation_file.url,
                                "key_file": aimodel.key_file,
                                "user": aimodel.user.id,
                                "extra": aimodel.extra,
                            },
                        },
                    )
                    response.raise_for_status()
                    response = response.json()
                except requests.exceptions.HTTPError as errh:
                    print("HTTP Error")
                    print(errh.args[0])
                    return Response(errh.args[0], status=response.status_code)
            else:
                result = AIModelPredictionTask(serializer.data, aimodel)
                pred_serializer = AIModelPredictionSerializer(result)
                response = pred_serializer.data
            workspace = Workspace.objects.filter(
                id=request.user.current_workspace
            ).first()
            if request.headers.get("APIKEY") is not None:
                api_key = get_object_or_404(APIKey, key=request.headers.get("APIKEY"))

                APILogger.objects.create(
                    uuid=uuid.uuid4().hex,
                    api_key=api_key,
                    model_type=SwitchModel(aimodel.model_type),
                    workspace=workspace,
                    data=input_url,
                    is_custom_model=True,
                    model_id=aimodel,
                )
            return Response(response, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class OCRTrainingView(APIView):
    def post(self, request):
        # Read the OCR_HOST URL from .env file or settings.py
        ocr_host = config("OCR_HOST")
        # Define the payload for the request
        user_data = {
            "origin": request.headers.get("Origin"),
            "email": request.data.get("email"),
            "username": request.data.get("username"),
            "model_type": request.data.get("model_type"),
            "annotation_file": request.data.get("annotation_file"),
            "aimodel_id": request.data.get("aimodel_id"),
            "aimodel_name": request.data.get("aimodel_name"),
            "uuid": request.data.get("uuid"),
        }
        # Make the API request to the new URL
        response = requests.post(f"{ocr_host}/api/aimodels/", json=user_data)
        # Check the response status code and handle any errors
        if response.status_code == 200:
            # Successful response
            return Response({"message": "OCR training started"})
        else:
            # Error response
            return Response(
                {"error": "Failed to start OCR training"}, status=response.status_code
            )


class OCRPredictionView(APIView):
    def post(self, request, pk):
        # Read the OCR_HOST URL from .env file or settings.py
        ocr_host = config("OCR_HOST")
        # Define the payload for the request
        user_data = {
            "aimodel": pk,
            "origin": request.headers.get("Origin"),
            "input_images": request.data.get("images"),
            "user": request.data.get("user"),
            "description": request.data.get("description"),
            "model_type": request.data.get("model_type"),
        }

        serializer = AIModelPredictionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()

        # Make the API request to the new URL
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{ocr_host}/api/aimodels/{pk}/prediction", json=user_data, headers=headers
        )
        # Check the response status code and handle any errors
        if response.status_code == 200:
            # Successful response
            response_data = response.json()
            return Response(response_data)
        else:
            # Error response
            return Response(
                {"error": "OCR prediction failed"}, status=response.status_code
            )


class SaveAIModelOCRPredictionData(APIView):
    def post(self, request, pk):
        serializer = AIModelPredictionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AIModelOCRPredictionDetails(APIView):
    def get_object(self, pk):
        try:
            return AIModelPrediction.objects.get(pk=pk)
        except AIModelPrediction.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        aimodel = self.get_object(pk)
        serializer = AIModelPredictionSerializer(aimodel)
        return Response(serializer.data)

    def put(self, request, pk):
        aimodel_prediction = self.get_object(pk)
        serializer = AIModelPredictionSerializer(
            aimodel_prediction, data=request.data, partial=True
        )

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AIModelAnnotationReTraining(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, pk):
        try:
            annotation_file = request.data.get("annotation_file")
        except KeyError:
            return Response(
                {"error": "please provide annotation file"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # which model we have to retrain
            model = AIModel.objects.get(id=pk)
        except AIModel.DoesNotExist:
            return Response(
                {"error": "please provide valid model id"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        previous_annotation_file = requests.get(model.annotation_file.url).json()
        previous_annotations = list(previous_annotation_file["annotations"])
        previous_images = list(previous_annotation_file["images"])

        new_annotation_file = json.load(annotation_file)
        new_annotations = previous_annotations + list(
            new_annotation_file["annotations"]
        )

        new_images = previous_images + list(new_annotation_file["images"])
        new_categories = list(new_annotation_file["categories"])

        updated_annotation_file = {
            "categories": new_categories,
            "annotations": new_annotations,
            "images": new_images,
        }
        # update new annotation file
        if request.data.get("type") != "append":
            model.annotation_file = annotation_file
            model.is_trained = False
            if request.data.get("name"):
                model.name = request.data.get("name")
            model.save()
        if request.data.get("type") == "append":
            if request.data.get("name"):
                model.name = request.data.get("name")
            model.annotation_file.save(
                annotation_file.name,
                ContentFile(
                    json.dumps(updated_annotation_file, indent=2).encode("utf-8")
                ),
                save=True,
            )

        user_data = {
            "origin": request.headers.get("Origin"),
            "email": request.user.email,
            "username": request.user.username,
        }

        django_rq.enqueue(
            AIModelTraining,
            model.uuid,
            user_data,
            job_timeout=18000,
            on_failure=failure_call_back,
        )

        return Response(
            {"message": "model retraining started"}, status=status.HTTP_200_OK
        )


class AIModelReTraining(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, pk):
        try:
            # which model we have to retrain
            model = AIModel.objects.get(id=pk)
        except AIModel.DoesNotExist:
            return Response(
                {"error": "please provide valid model id"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        model.name = request.data.get("name")
        model.is_trained = False
        model.save()

        user_data = {
            "origin": request.headers.get("Origin"),
            "email": request.user.email,
            "username": request.user.username,
        }

        django_rq.enqueue(
            AIModelTraining,
            model.uuid,
            user_data,
            job_timeout=18000,
            on_failure=failure_call_back,
        )

        return Response(
            {"message": "model retraining started"}, status=status.HTTP_200_OK
        )


class AnnotationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk, type):
        aimodel = AIModel.objects.filter(
            workspace=request.user.current_workspace,
            uuid=pk,
            model_type=request.query_params.get("model_type"),
        ).first()
        if aimodel is None:
            return throw_error("Model not found")
        serializer = AIModelSerializer(aimodel)
        data = dict(serializer.data)
        if (
            data["model_type"] == "classification"
            or data["model_type"] == "image_similarity"
        ):
            return throw_error("No annotation file")

        if (
            type == "categories"
            or type == "images"
            or type == "image"
            or type == "details"
        ) is False:
            return throw_error("Bad request")

        if data["annotation_file"]:
            annotation_data = requests.get(data["annotation_file"]).json()
        else:
            annotation_data = {"categories": [], "annotations": [], "images": []}

        annotated = []
        unannotated = []
        if type == "images" or type == "details":
            for img in annotation_data["images"]:
                if img["annotated"]:
                    annotated.append(img)
                else:
                    unannotated.append(img)
            if type == "details":
                image_id = ""
                if len(unannotated) > 0:
                    image_id = unannotated[0]["id"]
                elif len(annotated) > 0:
                    image_id = annotated[0]["id"]
                elif len(annotation_data["images"]) > 0:
                    image_id = annotation_data["images"][0]["id"]
                return Response(
                    {
                        "model_name": data["name"],
                        "annotated_count": len(annotated),
                        "unannotated_count": len(unannotated),
                        "id": data["id"],
                        "status": data["status"],
                        "image_id": image_id,
                    }
                )
            qury = annotation_data["images"]
            if request.query_params.get("unannotated"):
                qury = unannotated
            if request.query_params.get("annotated"):
                qury = annotated
            page = request.query_params.get("page", 1)
            per_page = request.query_params.get("per_page", 10)
            paginator = Paginator(qury, per_page)
            try:
                if int(page) > paginator.num_pages:
                    return Response(
                        {
                            "model_name": data["name"],
                            "pages": paginator.num_pages,
                            "images": [],
                            "annotated_count": len(annotated),
                            "unannotated_count": len(unannotated),
                            "status": data["status"],
                        },
                        status=status.HTTP_200_OK,
                    )
                chat = paginator.page(page)
            except PageNotAnInteger:
                chat = paginator.page(1)
            except EmptyPage:
                chat = paginator.page(paginator.num_pages)
            return Response(
                {
                    "model_name": data["name"],
                    "pages": paginator.num_pages,
                    "images": chat.object_list,
                    "annotated_count": len(annotated),
                    "unannotated_count": len(unannotated),
                    "status": data["status"],
                },
                status=status.HTTP_200_OK,
            )
        if type == "categories":
            result = list(annotation_data["categories"])
            result = [x for x in result if (str("deleted_") not in x.get("name"))]
            for cat in result:
                annotations = [
                    x
                    for x in annotation_data["annotations"]
                    if (cat["id"] == x.get("category_id"))
                ]
                cat["count"] = len(annotations)
            return Response(
                {
                    "model_name": data["name"],
                    "status": data["status"],
                    "categories": result,
                    "total": len(annotation_data["annotations"]),
                },
                status=status.HTTP_200_OK,
            )
        if type == "image":
            imgId = request.query_params.get("image_id")
            img_data = {"model_name": data["name"]}
            images_list = list(annotation_data["images"])
            for index, img in enumerate(images_list):
                img["index"] = index
                img["prev_id"] = images_list[index - 1]["id"]
                img["next_id"] = images_list[
                    0 if (index + 1 == len(images_list)) else index + 1
                ]["id"]
                if img["annotated"]:
                    annotated.append(img)
                else:
                    unannotated.append(img)
                if str(img["id"]) == imgId:
                    img_data.update(img)

            img_data.update({"annotated_count": len(annotated)})
            img_data.update({"unannotated_count": len(unannotated)})
            if imgId == "true":
                if len(unannotated) > 0:
                    img_data.update(unannotated[0])
                elif len(annotated) > 0:
                    img_data.update(annotated[0])

            if len(img_data.keys()) > 3:
                img_ano = get_annotations(img_data, annotation_data)
                img_ano["status"] = data["status"]
                return Response(img_ano, status=status.HTTP_200_OK)

        return throw_error("Bad request")

    def post(self, request, pk, type):
        # Validate pk parameter
        try:
            uuid.UUID(pk)
        except ValueError:
            return Response({"error": "Invalid pk parameter."}, status=400)

        aimodel = AIModel.objects.filter(
            workspace=request.user.current_workspace,
            uuid=pk,
            model_type=request.query_params.get("model_type"),
        ).first()
        if aimodel is None:
            return Response({"error": "Model not found."}, status=404)
        serializer = AIModelSerializer(aimodel)
        data = dict(serializer.data)

        # Validate type parameter
        valid_types = ["name", "category", "images", "file", "annotation"]
        if type not in valid_types:
            return Response({"error": "Invalid type parameter."}, status=400)
        if type == "name" and request.data.get("name"):
            aimodel.name = request.data.get("name")
            aimodel.save()
            return Response({"msg": "updated successfully"})

        if (
            data["model_type"] == "classification"
            or data["model_type"] == "image_similarity"
        ):
            return Response({"error": "No annotation file."}, status=400)

        if data["annotation_file"] is None and (
            type == "category" or type == "annotation"
        ):
            return Response({"error": "No annotation file."}, status=400)

        # Validate annotation_file field
        annotation_file = request.data.get("annotation_file")
        if annotation_file is None:
            return Response({"error": "Missing annotation_file field."}, status=400)
        try:
            new_annotation_data = json.loads(annotation_file.read())
        except json.JSONDecodeError:
            return Response({"error": "Invalid annotation_file field."}, status=400)

        annotation_data = post_annotations_data(
            data["annotation_file"], new_annotation_data, type
        )
        if annotation_data["err"]:
            return Response({"error": annotation_data["err"]}, status=400)
        else:
            del annotation_data["err"]
            msg = annotation_data["success_msg"]
            del annotation_data["success_msg"]
            if request.data.get("name"):
                aimodel.name = request.data.get("name")

            aimodel.annotation_file.save(
                "Annotate.json",
                ContentFile(json.dumps(annotation_data, indent=2).encode("utf-8")),
                save=True,
            )
            aimodel.save()
            return Response({"msg": msg})

    def put(self, request, pk, type):
        aimodel = AIModel.objects.filter(
            workspace=request.user.current_workspace,
            uuid=pk,
            model_type=request.query_params.get("model_type"),
        ).first()
        if aimodel is None:
            return throw_error("Model not found")
        serializer = AIModelSerializer(aimodel)
        data = dict(serializer.data)
        if (
            data["model_type"] == "classification"
            or data["model_type"] == "image_similarity"
        ):
            return throw_error("No annotation file")

        if (type == "category" or type == "annotation" or type == "details") is False:
            return throw_error("Bad request")
        if type == "details" and request.data.get("name"):
            aimodel.save()
            return Response({"msg": "updated successfully"})

        if data["annotation_file"] is None and (
            type == "category" or type == "annotation"
        ):
            return throw_error("No annotation file")
        try:
            new_annotation_data = json.load(request.data.get("annotation_file"))
        except:
            return throw_error("required field annotation_file")

        annotation_data = update_annotations_data(
            data["annotation_file"], new_annotation_data, type
        )
        if annotation_data["err"]:
            return throw_error(annotation_data["err"])
        else:
            del annotation_data["err"]
            if request.data.get("name"):
                aimodel.name = request.data.get("name")

            aimodel.annotation_file.save(
                "Annotate.json",
                ContentFile(json.dumps(annotation_data, indent=2).encode("utf-8")),
                save=True,
            )
            aimodel.save()
            return Response({"msg": "updated successfully"})

    def delete(self, request, pk, type):
        annotator_permissions(self, request)
        aimodel = AIModel.objects.filter(
            workspace=request.user.current_workspace,
            uuid=pk,
            model_type=request.query_params.get("model_type"),
        ).first()
        if aimodel is None:
            return throw_error("Model not found")
        serializer = AIModelSerializer(aimodel)
        data = dict(serializer.data)
        if (
            data["model_type"] == "classification"
            or data["model_type"] == "image_similarity"
        ):
            return throw_error("No annotation file")

        delete_id = request.query_params.get("id")
        if (
            type == "image" or type == "annotation" or type == "category"
        ) is False or delete_id is None:
            return throw_error("Bad request")
        if data["annotation_file"] is None:
            return throw_error("No annotation file")
        if type == "images":
            annotator_permissions(self, request)
        annotation_data = delete_annotations_data(
            data["annotation_file"], type, delete_id
        )

        msg = annotation_data["success_msg"]

        del annotation_data["success_msg"]
        aimodel.annotation_file.save(
            "Annotate.json",
            ContentFile(json.dumps(annotation_data, indent=2).encode("utf-8")),
            save=True,
        )
        aimodel.save()
        return Response({"msg": msg})
