import os
import gc
import uuid
import json
import warnings
import cv2
import requests
import torch
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, nms


from PIL import Image
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from moviepy.editor import VideoFileClip

from deeplobe_api.api.views import save_frame_in_s3

warnings.filterwarnings("ignore")


def getAnnIds(data_json, image_id):
    return [
        data_dict["id"]
        for data_dict in data_json["annotations"]
        if image_id == data_dict["image_id"]
    ]


def loadAnns(data_json, ann_ids):
    annotations = []
    for ann_id in ann_ids:
        for data_dict in data_json["annotations"]:
            if ann_id == data_dict["id"]:
                annotations.append(data_dict)
    return annotations


def loadImgs(data_json, image_id):
    return [
        data_dict for data_dict in data_json["images"] if data_dict["id"] == image_id
    ]


class UserDataset(torch.utils.data.Dataset):
    def __init__(self, annotation, transforms=None, request_uuid=None):
        self.annotation = annotation
        self.transforms = transforms
        self.ids = sorted([data_dict["id"] for data_dict in annotation["images"]])
        self.request_uuid = request_uuid

    def __getitem__(self, index: int):
        image_id = self.ids[index]  # image id
        annotation_ids = getAnnIds(
            self.annotation, image_id
        )  # List: get annotation id from coco
        coco_annotation = loadAnns(
            self.annotation, annotation_ids
        )  # Dictionary: target coco_annotation file for an image
        image_path = loadImgs(self.annotation, image_id)[0][
            "file_name"
        ]  # path for input image
        image_url = loadImgs(self.annotation, image_id)[0]["url"]
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        number_of_objects = len(coco_annotation)  # number of objects in the image

        boxes = []
        labels = []
        for index in range(number_of_objects):
            xmin = int(coco_annotation[index]["bbox"][0])
            ymin = int(coco_annotation[index]["bbox"][1])
            xmax = int(xmin + coco_annotation[index]["bbox"][2])
            ymax = int(ymin + coco_annotation[index]["bbox"][3])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[index]["category_id"] + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([image_id])  # Tensorise image_id
        labels = torch.tensor(labels)

        my_annotation = {}  # Annotation is in dictionary format
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = image_id
        if self.transforms is not None:
            image = self.transforms(image)
        return image, my_annotation

    def __len__(self):
        return len(self.ids)


def get_class_dict(coco_instances):
    categories = coco_instances["categories"]
    class_dict = {dictionary["id"] + 1: dictionary["name"] for dictionary in categories}
    num_classes = 1 + len(class_dict)  # Number of categories and a background
    return num_classes, class_dict


def collate_fn(batch):
    return tuple(zip(*batch))


class datapreprocess(pl.LightningDataModule):
    def __init__(self, dataset, valid_pct=0.2, batch_size=2):
        super().__init__()
        self.coco_instances = dataset
        self.batch_size = batch_size
        self.valid_pct = valid_pct

        self.transforms = T.Compose([T.ToTensor()])
        self.prepare_data()

    def prepare_data(self):
        self.user_dataset = UserDataset(
            annotation=self.coco_instances, transforms=self.transforms
        )
        self.datalength = len(self.user_dataset)

        trl, tel = int(np.floor(self.datalength * 0.8)), int(
            np.ceil(self.datalength * 0.1)
        )
        vel = self.datalength - (trl + tel)
        print(
            "train data length :",
            trl,
            "\ntest data length :",
            tel,
            "\nvalid data length :",
            vel,
        )
        if vel == 0:
            print("test and valid dataset are same due to low data")
            (self.train_set, self.test_set) = random_split(
                self.user_dataset, [trl, tel]
            )
            self.val_set = self.test_set
        else:
            (self.train_set, self.test_set, self.val_set) = random_split(
                self.user_dataset, [trl, tel, vel]
            )

    def train_dataloader(
        self,
    ):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(
        self,
    ):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def val_dataloader(
        self,
    ):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model"""
    if pred.shape[0] == 0:
        print("no box detected, Hence 0 IOU below")
        return torch.tensor(0.0, device=pred.device)
    return box_iou(target["boxes"], pred).diag().mean()


class Taggermodel(pl.LightningModule):
    def __init__(self, classes):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = classes
        self.lr = 0.005
        self.momentum = 0.01

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            self.in_features, self.num_classes
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        imagetensors, annotationsdict = batch
        images = list(image for image in imagetensors)
        annotations = [
            {key: value for key, value in annotation.items()}
            for annotation in annotationsdict
        ]
        output = self.model(images, annotations)
        losses = sum(loss for loss in output.values())
        self.log("train_loss", losses)
        return losses

    def validation_step(self, batch, batch_idx):
        imagetensors, annotationsdict = batch
        images = list(image for image in imagetensors)
        annotations = [
            {key: value for key, value in annotation.items()}
            for annotation in annotationsdict
        ]
        output = self.model(images, annotations)

        nms_indices = nms(
            boxes=output[0]["boxes"], scores=output[0]["scores"], iou_threshold=0.1
        )
        pred_boxes = [output[0]["boxes"][idx] for idx in nms_indices]
        try:
            print("inside try in validation")
            pred_boxes = torch.stack(pred_boxes)
        except:
            print("inside except in validation")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pred_boxes = torch.stack([torch.tensor([0, 0, 0, 0]).to(device)])
        iou = torch.stack(
            [_evaluate_iou(t, o.unsqueeze(0)) for t, o in zip(annotations, pred_boxes)]
        ).mean()
        self.log("val_loss", iou)
        return iou

    def test_step(self, batch, batch_idx):
        imagetensors, annotationsdict = batch
        images = list(image for image in imagetensors)

        annotations = [
            {key: value for key, value in annotation.items()}
            for annotation in annotationsdict
        ]

        outs = self.model(images)
        nms_indices = nms(
            boxes=outs[0]["boxes"], scores=outs[0]["scores"], iou_threshold=0.1
        )
        pred_boxes = [outs[0]["boxes"][idx] for idx in nms_indices]
        try:
            print("inside try in validation")
            pred_boxes = torch.stack(pred_boxes)
        except:
            print("inside except in validation")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pred_boxes = torch.stack([torch.tensor([0, 0, 0, 0]).to(device)])
        self.iou = torch.stack(
            [_evaluate_iou(t, o.unsqueeze(0)) for t, o in zip(annotations, pred_boxes)]
        ).mean()
        return {"test_iou": self.iou}

    def test_epoch_end(self, outs):
        avg_iou = torch.stack([o["test_iou"] for o in outs]).mean()
        self.log("test_iou", self.iou)
        self.log("avg_test_iou", avg_iou)
        # return {"avg_test_iou": avg_iou, "log": self.iou}
        return

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum
        )
        return optimizer


class Objecttagger:
    def load_data(self, annotation_json_path, download_path):
        r = requests.get(annotation_json_path)
        self.download_path = download_path
        self.data_json = json.loads(r.text)
        (self.classes, self.classdict), self.data = (
            get_class_dict(self.data_json),
            datapreprocess(self.data_json),
        )

    def train(self, mode="production"):
        model = Taggermodel(self.classes)
        print("model training started")
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=self.download_path
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=5, verbose=False, mode="min", min_delta=0.05
        )
        devices = torch.cuda.device_count() if torch.cuda.device_count() != 0 else 1
        epochs = 200 if mode == "production" else 5 if mode == "staging" else 1
        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir=self.download_path,
            callbacks=[self.checkpoint_callback, early_stop_callback],
            devices=devices,
            accelerator="auto",
            benchmark=True,
        )
        trainer.fit(model, self.data)
        print("testing the model now")
        result = trainer.test(datamodule=self.data)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return result


class TaggerPredict:
    def __init__(self, weight_file, annotation_file):
        f = requests.get(annotation_file)
        data_json = json.loads(f.text)

        categories = data_json["categories"]
        self.class_dict = {categ["id"] + 1: categ["name"] for categ in categories}
        self.model = Taggermodel(len(categories))
        self.model = self.model.load_from_checkpoint(weight_file)
        self.model = self.model.eval()

    def plot_output(self, input_image, threshold=None, is_image_file=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda()
        transform = T.Compose([T.ToTensor()])
        tranformed_img = transform(input_image).to(device)
        loaded_img = self.model([tranformed_img])
        width, height = input_image.size

        nms_indices = nms(
            boxes=loaded_img[0]["boxes"],
            scores=loaded_img[0]["scores"],
            iou_threshold=0.1,
        )

        prediction_score = [
            loaded_img[0]["scores"][idx].detach().cpu() for idx in nms_indices
        ]

        pred_boxes = [
            loaded_img[0]["boxes"][idx].detach().cpu().numpy() for idx in nms_indices
        ]
        prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in pred_boxes]

        pred_class = [
            loaded_img[0]["labels"][idx].detach().cpu().numpy() for idx in nms_indices
        ]
        prediction_class = [self.class_dict[int(label)] for label in pred_class]

        filtered_indices = [
            index for index, score in enumerate(prediction_score) if score > threshold
        ]

        image_array = np.array(input_image)

        fontScale = 1
        thickness = 2

        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        colours = []
        bbox = []
        labels = []
        scores = []

        for index in filtered_indices:
            color_list = np.random.choice(range(256), size=3).tolist()

            cords = tuple(tuple(map(int, tup)) for tup in prediction_boxes[index])

            cv2.rectangle(
                image_array, cords[0], cords[1], color_list[::-1], 3
            )  # for image

            x_min, y_min = prediction_boxes[index][0]

            category_name = prediction_class[index]

            colours.append(color_list)
            bbox.append(cords)
            labels.append(category_name)
            scores.append(prediction_score[index].item())

            ((text_width, text_height), _) = cv2.getTextSize(
                category_name, cv2.FONT_HERSHEY_DUPLEX, fontScale, thickness
            )
            cv2.rectangle(
                image_array,
                (int(x_min), int(y_min) - int(1.3 * text_height)),
                (int(x_min + text_width), int(y_min)),
                # (0, 240, 0),
                (255, 255, 255),
                -1,
            )  # for label

            image_array = cv2.putText(
                image_array,
                category_name,
                (int(x_min), int(y_min) - int(0.2 * text_height)),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=fontScale,
                # color=(0, 0, 240),
                color=(0, 0, 0),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        image_array = cv2.resize(image_array, (width, height))

        if is_image_file:
            return image_array, labels, scores, bbox, colours

        else:
            return image_array

    def predict_on_image(self, input_image, output_image, aimodel):
        uid = uuid.uuid4()
        output_image = f"{output_image}/{uid}.jpg"
        input_image = Image.open(input_image).convert("RGB")
        image_array, labels, scores, bbox, colours = self.plot_output(
            input_image, threshold=0.7, is_image_file=True
        )
        cv2.imwrite(output_image, image_array)

        props = []
        for i in range(len(labels)):
            d = {}
            try:
                d["label"] = labels[i]
            except:
                d["label"] = ""

            try:
                d["predicted_score"] = scores[i]
            except:
                d["predicted_score"] = ""

            try:
                d[
                    "coordinates"
                ] = f"[({bbox[i][0][0]}, {bbox[i][0][1]+bbox[i][1][1]}), ({bbox[i][0][0]+bbox[i][1][0]}, {bbox[i][0][1]+bbox[i][1][1]}), ({bbox[i][0][0]+bbox[i][1][0]}, {bbox[i][0][1]}), ({bbox[i][0][0]}, {bbox[i][0][1]})]"
            except:
                d["coordinates"] = ""

            try:
                d["color"] = f"rgb({colours[i][0]},{colours[i][1]},{colours[i][2]})"
            except:
                d["color"] = ""
            props.append(d)

        body = open(output_image, "rb")
        url = save_frame_in_s3(
            f"object_detection/{aimodel}/predictions/{uid}.jpg", body, "image/jpg"
        )
        os.remove(output_image)

        return {"image": url, "json": props}

    def predict_on_video(self, input_video, output_video, uid, aimodel, fps=30):
        out_video = f"{output_video}/{uid}_masked.mp4"
        video_capture = cv2.VideoCapture(input_video)

        set_audio = False if fps < int(video_capture.get(cv2.CAP_PROP_FPS)) else True

        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")

        output_file = cv2.VideoWriter(
            out_video, fourcc, fps, (frame_width, frame_height), True
        )
        a = 0
        while True:
            cap, frame = video_capture.read()
            if cap == False:
                break
            a += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(rgb_frame)

            image_array = self.plot_output(
                input_image, threshold=0.7, is_image_file=False
            )
            output_file.write(image_array)
        output_file.release()
        video_capture.release()
        final_video = f"{output_video}/{uid}-final.mp4"
        if set_audio:
            self.audio_joiner(input_video, out_video, final_video)
        body = open(final_video, "rb")
        url = save_frame_in_s3(
            f"object_detection/{aimodel}/predictions/{uid}.mp4", body, "video/mp4"
        )
        os.remove(input_video)
        os.remove(out_video)
        os.remove(final_video)

        return url

    def audio_joiner(self, input_video, masked_video, output_video):
        # Load the video file
        video = VideoFileClip(input_video)

        # Extract the audio
        audio = video.audio

        # # Load the video file
        video = VideoFileClip(masked_video)

        # Set the audio duration to match the video duration
        audio = audio.set_duration(video.duration) if audio else None

        # Add the audio to the video
        result = video.set_audio(audio)  # Write the result to a file
        result.write_videofile(output_video)
