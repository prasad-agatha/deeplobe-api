import json
import requests

from rest_framework import status
from rest_framework.response import Response

from django.core.files.base import ContentFile

from deeplobe_api.db.models import AIModel, UserInvitation
from deeplobe_api.utils.emails import emails


def throw_error(msg):
    return Response({"msg": msg}, status=status.HTTP_400_BAD_REQUEST)


def get_annotations(img, annotation_data):
    annotations = []
    for annotation in annotation_data["annotations"]:
        if ("image_id" in annotation.keys()) and (annotation["image_id"] == img["id"]):
            annotations.append(annotation)
    img["annotations"] = annotations
    return img


def SwitchModel(model_type):
    switcher = {
        "classification": "Custom Classification",
        "segmentation": "Custom Segmentation",
        "image_similarity": "Custom Image Similarity",
        "object_detection": "Custom Object detection",
        "instance": "Custom Instance",
        "ocr": "Custom OCR",
        "llm": "Custom LLM"
    }
    return switcher.get(model_type, "Invalid model type")


def get_user_role(request):
    collaborator_user = UserInvitation.objects.filter(
        workspace=request.user.current_workspace,
        collaborator_email=request.user.email,
        is_collaborator=True,
    ).first()
    result = {
        "role": "owner",
        "models": {
            "select": ["All"],
            "api": True,
            "create_model": True,
            "delete_model": True,
            "test_model": True,
        },
    }
    if collaborator_user:
        result["role"] = collaborator_user.role
        result["models"] = collaborator_user.models[0]
    return result


def check_model_name_availability(request, name):
    """
    Check if an AIModel with the given name already exists in the current workspace.
    """
    if AIModel.objects.filter(
        name=name, workspace=request.user.current_workspace
    ).exists():
        return f"Model name '{name}' already exists."
    return None


def failure_call_back(job, *args, **kwargs):
    aimodel_id, data = job.args
    aimodel = AIModel.objects.filter(uuid=aimodel_id).first()
    aimodel.is_failed = True
    aimodel.failed_reason = {"reason": job.exc_info}
    model_class = SwitchModel(aimodel.model_type)
    if aimodel.model_type == "image_similarity":
        model_type = "similarity"
    else:
        model_type = aimodel.model_type
    emails.custom_model_failed_email(aimodel, data, model_class, model_type)
    aimodel.save()
    aimodel.delete()


def update_annotation_file(request, model, annotation_file):
    """
    Update an AIModel's annotation file with the contents of the given annotation_file.
    """
    previous_annotation_file = json.loads(model.annotation_file.read().decode("utf-8"))
    previous_annotations = list(previous_annotation_file["annotations"])
    previous_images = list(previous_annotation_file["images"])

    new_annotation_file = json.load(annotation_file)
    new_annotations = previous_annotations + list(new_annotation_file["annotations"])
    new_images = previous_images + list(new_annotation_file["images"])
    new_categories = list(new_annotation_file["categories"])

    updated_annotation_file = {
        "categories": new_categories,
        "annotations": new_annotations,
        "images": new_images,
    }

    # update new annotation file
    if request.data.get("type") != "append":
        if request.data.get("name"):
            model.name = request.data.get("name")
        model.annotation_file.delete()  # delete previous annotation file
        model.annotation_file.save(  # save new annotation file
            annotation_file.name,
            ContentFile(json.dumps(updated_annotation_file, indent=2).encode("utf-8")),
            save=True,
        )
        model.is_trained = False
        model.status = "Re-Train"
        model.save()

    if request.data.get("type") == "append":
        if request.data.get("name"):
            model.name = request.data.get("name")
        model.is_trained = False
        model.status = "Re-Train"
        model.annotation_file.save(
            annotation_file.name,
            ContentFile(json.dumps(updated_annotation_file, indent=2).encode("utf-8")),
            save=True,
        )
        model.save()


import requests


def post_annotations_data(url, new_annotation_data, type):
    err = ""
    if type == "file":
        # TODO check file format
        categories = list(new_annotation_data["categories"])
        annotations = list(new_annotation_data["annotations"])
        images = list(new_annotation_data["images"])
        success_msg = "uploaded successfully"
        return {
            "categories": categories,
            "annotations": annotations,
            "images": images,
            "err": err,
            "success_msg": success_msg,
        }

    if url is None:
        return {
            "categories": [],
            "annotations": [],
            "images": list(new_annotation_data["images"]),
            "err": err,
            "success_msg": "uploaded successfully",
        }

    annotation_file = requests.get(url).json()
    # TODO check image object formats and remove duplicate ids
    categories = list(annotation_file["categories"])
    annotations = list(annotation_file["annotations"])
    images = list(annotation_file["images"])
    # TODO check for unique ids

    if type == "category":
        new_cat = dict(new_annotation_data["categories"][0])
        # Check err
        exists_cat = [x for x in categories if (new_cat["name"] == x.get("name"))]
        success_msg = "uploaded successfully"
        if len(exists_cat) > 0:
            err = "Label already exists"
        else:
            new_id = len(categories)
            success_msg = {"id": new_id}
            new_cat.update({"id": new_id})
            categories += list([new_cat])

    if type == "annotation":
        new_anns = list(new_annotation_data["annotations"])
        img_id = new_annotation_data["image_id"]
        # Validate image_id
        if not img_id:
            err = "image_id cannot be empty"
            return {
                "categories": categories,
                "annotations": annotations,
                "images": images,
                "err": err,
                "success_msg": success_msg,
            }
        if not any(img_id == img["id"] for img in images):
            err = "Invalid image_id"
            return {
                "categories": categories,
                "annotations": annotations,
                "images": images,
                "err": err,
                "success_msg": success_msg,
            }
        annotations = [x for x in annotations if (img_id != x.get("image_id"))]
        for img in images:
            if img["id"] == img_id:
                annotations += list(new_anns)
                img.update({"annotated": True if len(list(new_anns)) > 0 else False})
        success_msg = "uploaded successfully"

    if type == "images":
        # TODO check new_annotation_data["images"] with images list and remove duplicate ids
        new_images = list(new_annotation_data["images"])
        for new_img in new_images:
            # Validate id
            if not new_img["id"]:
                err = "id cannot be empty"
                return {
                    "categories": categories,
                    "annotations": annotations,
                    "images": images,
                    "err": err,
                    "success_msg": success_msg,
                }
            if any(new_img["id"] == img["id"] for img in images):
                err = "Duplicate image id"
                return {
                    "categories": categories,
                    "annotations": annotations,
                    "images": images,
                    "err": err,
                    "success_msg": success_msg,
                }
            images.append(new_img)
        success_msg = "uploaded successfully"

    return {
        "categories": categories,
        "annotations": annotations,
        "images": images,
        "err": err,
        "success_msg": success_msg,
    }


def update_annotations_data(url, new_annotation_data, type):
    err = ""
    annotation_file = requests.get(url).json()
    # TODO check image object formats and remove duplicate ids
    categories = list(annotation_file["categories"])
    annotations = list(annotation_file["annotations"])
    images = list(annotation_file["images"])
    # TODO check for unique ids
    if type == "category":
        upd_cat = dict(new_annotation_data["categories"][0])
        # Check err
        upd_exists_cat = [
            x
            for x in categories
            if ((upd_cat["id"] == x.get("id")) or upd_cat["name"] == x.get("name"))
        ]
        if len(upd_exists_cat) == 1:
            for cat in categories:
                if cat["id"] == upd_cat["id"]:
                    cat.update(upd_cat)
        elif len(upd_exists_cat) > 1:
            err = "Label already exists"
        else:
            err = "Label does not exists"

    if type == "annotation":
        upd_ann = dict(new_annotation_data["annotations"][0])
        # TODO check if image_id and category_id of new_ann is present in images and categories list
        # if not update err and dont add new_ann to annotations
        upd_exists_ann = [x for x in annotations if (upd_ann["id"] == x.get("id"))]
        if len(upd_exists_ann) > 0:
            for ann in annotations:
                if ann["id"] == upd_ann["id"]:
                    ann.update(upd_ann)
        else:
            err = "Annotation does not exists"

    return {
        "categories": categories,
        "annotations": annotations,
        "images": images,
        "err": err,
    }


def delete_annotations_data(url, type, delete_id):
    annotation_file = requests.get(url).json()
    # TODO check image object formats and remove duplicate ids
    categories = list(annotation_file["categories"])
    annotations = list(annotation_file["annotations"])
    images = list(annotation_file["images"])
    # TODO check for unique ids
    if type == "image":
        # remove image and related annotations
        images = [x for x in images if not (delete_id == str(x.get("id")))]
        annotations = [
            x for x in annotations if not (delete_id == str(x.get("image_id")))
        ]
        success_msg = "deleted successfully"
    elif type == "annotation":
        # remove related annotations
        new_ann = []
        del_ann = {}
        for ann in annotations:
            if str(ann["id"]) == delete_id:
                del_ann = dict(ann)
            else:
                new_ann.append(ann)
        annotations = list(new_ann)
        other_ann_on_img = [
            x for x in annotations if (del_ann["image_id"] == x.get("image_id"))
        ]
        unannotated_count = 0
        for img in images:
            if img["id"] == del_ann["image_id"] and len(other_ann_on_img) == 0:
                unannotated_count = unannotated_count + 1
                img.update({"annotated": False})
            elif img["annotated"] is False:
                unannotated_count = unannotated_count + 1
        success_msg = {
            "annotated_count": len(images) - unannotated_count,
            "unannotated_count": unannotated_count,
        }
    elif type == "category":
        # remove category and related annotations
        for cat in categories:
            if str(cat.get("id")) == delete_id:
                upd_cat = dict(cat)
                upd_cat["name"] = "deleted_" + upd_cat["name"]
                cat.update(upd_cat)
        annotations = [
            x for x in annotations if not (delete_id == str(x.get("category_id")))
        ]
        for img in images:
            ann_on_img = [
            x for x in annotations if (img["id"] == x.get("image_id"))
        ]
            upd_img = dict(img)
            upd_img["annotated"] = True  if len(ann_on_img)>0 else False
            img.update(upd_img)
        success_msg = "deleted successfully"
    else:
        return throw_error("Bad request")

    return {
        "categories": categories,
        "annotations": annotations,
        "images": images,
        "success_msg": success_msg,
    }
