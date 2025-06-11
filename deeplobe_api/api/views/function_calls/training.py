import os
import wget
from decouple import config
from django.conf import settings

from deeplobe_ai.semantic import Semantic
from deeplobe_ai.instance import Instance
from deeplobe_ai.tagger import Objecttagger
from deeplobe_ai.similarity import Imagesimilarity
from deeplobe_ai.classification import Classification
from deeplobe_ai.llm import LLM
from deeplobe_api.utils.emails import emails
from deeplobe_api.db.models import AIModel, MediaAsset
from deeplobe_api.api.views.function_calls.aimodels import SwitchModel

from deeplobe_api.api.views.aws import save_frame_in_s3,upload_folder_to_s3


BASE_DIR = settings.BASE_DIR


def AIModelTraining(uuid, request):
    aimodel = AIModel.objects.get(uuid=uuid)
    save_model_path = f"{BASE_DIR}/data/{uuid}/"
    print(save_model_path,"path")
    # os.makedirs(save_model_path, exist_ok=True)
    mode = request.get("training_mode", "staging")

    if aimodel.model_type == "object_detection":
        detection_model = Objecttagger()
        detection_model.load_data(
            aimodel.annotation_file.url, save_model_path
        )  # A folder with images folder and annotation folder in it
        # result = detection_model.train(uuid, epochs=1)  # Change epochs in train

        result = detection_model.train(mode=mode)
        aimodel.extra = result
        aimodel.is_trained = True
        aimodel.status = "Live"
        aimodel.save()

    if aimodel.model_type == "segmentation":
        semseg_model = Semantic()
        semseg_model.load_data(aimodel.annotation_file.url, save_model_path)
        # result = semseg_model.train(1)

        result = semseg_model.train(mode=mode)
        aimodel.extra = result
        aimodel.is_trained = True
        aimodel.status = "Live"
        aimodel.save()

    if aimodel.model_type == "classification":
        classification_media = MediaAsset.objects.filter(name=uuid)
        cl_model = Classification()
        preprocess = {
            "auto_orient": True,
            "resize": False,
            "output_size": "",
            "fit_method": "",
        }
        if aimodel.preprocessing:
            preprocess = {
                "auto_orient": True if not aimodel.preprocessing or not aimodel.preprocessing.get("resize") else aimodel.preprocessing.get("auto_orient"),
                "resize": False if not aimodel.preprocessing else aimodel.preprocessing.get("resize"),
                "output_size": "" if not aimodel.preprocessing or not aimodel.preprocessing.get("resize") else None if aimodel.preprocessing.get("auto_orient") else tuple(aimodel.preprocessing.get("output_size")),
                "fit_method": "" if not aimodel.preprocessing or not aimodel.preprocessing.get("resize") else None if aimodel.preprocessing.get("auto_orient") else aimodel.preprocessing.get("fit_method"),
            }
        cl_model.load_data(
            url_dict=classification_media,
            save_model_loc=save_model_path,
            auto_orient_bool=preprocess["auto_orient"],
            custom_resize_bool=preprocess["resize"],
            output_size=preprocess["output_size"],
            fit_method=preprocess["fit_method"],
        )
        # cl_model.load_data(classification_media, save_model_path)
        # result = cl_model.train(epochs=1)

        result = cl_model.train(mode=mode)
        aimodel.extra = result
        aimodel.is_trained = True
        aimodel.status = "Live"
        aimodel.save()

    if aimodel.model_type == "image_similarity":
        similarity_media = MediaAsset.objects.filter(name=uuid)
        similarity_model = Imagesimilarity()
        preprocess = {
            "auto_orient": True,
            "resize": False,
            "output_size": "",
            "fit_method": "",
        }

        if aimodel.preprocessing:
           preprocess = {
                "auto_orient": True if not aimodel.preprocessing or not aimodel.preprocessing.get("resize") else aimodel.preprocessing.get("auto_orient"),
                "resize": False if not aimodel.preprocessing else aimodel.preprocessing.get("resize"),
                "output_size": "" if not aimodel.preprocessing or not aimodel.preprocessing.get("resize") else None if aimodel.preprocessing.get("auto_orient") else tuple(aimodel.preprocessing.get("output_size")),
                "fit_method": "" if not aimodel.preprocessing or not aimodel.preprocessing.get("resize") else None if aimodel.preprocessing.get("auto_orient") else aimodel.preprocessing.get("fit_method"),
            }
        similarity_model.load_data(
            similarity_media,
            save_model_path,
            preprocess["auto_orient"],
            preprocess["resize"],
            preprocess["output_size"],
            preprocess["fit_method"],
        )
        # similarity_model.load_data(similarity_media, save_model_path)
        # result = similarity_model.train(epochs=1)  # default epochs =  2

        result = similarity_model.train(mode=mode)
        aimodel.extra = result
        aimodel.is_trained = True
        aimodel.status = "Live"
        aimodel.save()

    if aimodel.model_type == "instance":
        instanceseg_model = Instance()
        instanceseg_model.load_data(aimodel.annotation_file.url, save_model_path)
        # result = instanceseg_model.train(
        #     epochs=1
        # )  # default epochs = 10 if not mentioned

        result = instanceseg_model.train(mode=mode)
        aimodel.extra = result
        aimodel.is_trained = True
        aimodel.status = "Live"
        aimodel.save()
    
    if aimodel.model_type == "llm":
        llm_media = MediaAsset.objects.filter(name=uuid)
        llm_model = LLM(save_model_path)  # Use the LLM class, not the Classifier class
        docs = llm_model.create_docs(llm_media)
        result = llm_model.store_embeddings(docs)
        if result:
            aimodel.is_trained = True
            aimodel.status = "Live"
            aimodel.save()

   

    for file in os.listdir(save_model_path):
        if os.path.isdir(os.path.join(save_model_path, file)) and file == "vectorDB":
            local_file = os.path.join(save_model_path, file)
            # Upload the "vectorDB" folder to S3 and get all the uploaded URLs
            bucket_name = config("AWS_S3_BUCKET_NAME")
            upload_folder_to_s3(local_file, bucket_name)
            # After processing the folder, send the email for the "vectorDB" folder
            model_class = SwitchModel(aimodel.model_type)
            if aimodel.model_type == "image_similarity":
                model_type = "similarity"
            elif aimodel.model_type == "object_detection":
                model_type = "object-detection"
            else:
                model_type = aimodel.model_type

            # Send the custom model trained email for the "vectorDB" folder with all the uploaded URLs
            emails.custom_model_trained_email(request, aimodel, model_class, model_type)


        else:
            if file.endswith((".ckpt", ".bin")):
                # Existing file processing logic (same as before)
                local_file = os.path.join(save_model_path, file)
                body = open(local_file, "rb")
                print("ckpt file upload started")
                aws_url = save_frame_in_s3(
                    f"{aimodel.model_type}/{aimodel.uuid}/{file}", body
                )
                print("file uploaded successfully.")
                url = f"data/{uuid}/{file}"
                aimodel.weight_file = url
                aimodel.aws_weight_file = aws_url
                aimodel.save()

                model_class = SwitchModel(aimodel.model_type)
                if aimodel.model_type == "image_similarity":
                    model_type = "similarity"
                elif aimodel.model_type == "object_detection":
                    model_type = "object-detection"
                else:
                    model_type = aimodel.model_type

                emails.custom_model_trained_email(request, aimodel, model_class, model_type)

            else:
                print(f"Skipping file/folder: {file}")




