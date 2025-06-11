import cv2
import os
import gc
import json
from PIL import Image
import numpy as np

import requests

import torch
import torch.nn as nn

from torch.utils.data import random_split, DataLoader


from torchvision import transforms
from torchvision.models.segmentation.fcn import FCNHead
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import models
import torchvision.transforms as T

from pycocotools.coco import COCO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class CustomDataset:
    def __init__(self, ann_path):
        self.ann_file = ann_path
        self.coco = COCO(self.ann_file)

        catIDs = self.coco.getCatIds()
        self.categories = self.coco.loadCats(catIDs)
        self.class_dict = {
            dictionary["id"] + 1: dictionary["name"] for dictionary in self.categories
        }
        self.class_dict[0] = "background"
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        """mask conversion
        self.mask_transform = T.Compose([T.Resize((256,256)), T.ToTensor()])"""

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]  # image id
        annotation_ids = self.coco.getAnnIds(
            imgIds=image_id
        )  # List: get annotation id from coco
        coco_annotation = self.coco.loadAnns(
            annotation_ids
        )  # Dictionary: target coco_annotation file for an image
        image_annotation = self.coco.loadImgs(image_id)[0]  # path for input image

        image_url = self.coco.loadImgs(image_id)[0]["url"]
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        image_annotation["height"] = image.size[1]
        image_annotation["width"] = image.size[0]

        image1 = self.img_transform(image)

        for index, annotation in enumerate(coco_annotation):
            mask = annotation["segmentation"]
            mask = [[point for points in mask for point in points]]
            coco_annotation[index]["segmentation"] = mask

        dummy = np.zeros((image.size[1], image.size[0]))
        for i, ann_single in enumerate(coco_annotation):
            dummy = np.maximum(
                dummy, self.coco.annToMask(ann_single) * [ann_single["category_id"] + 1]
            )
        dummy = cv2.resize(dummy, (256, 256))
        mask = torch.as_tensor(dummy, dtype=torch.uint8).unsqueeze(0).long()

        return image1, mask


class DataModulesemantic(pl.LightningDataModule):
    def __init__(self, json_path):
        super().__init__()
        self.json_path = json_path
        self.train_dataset = CustomDataset(self.json_path)
        self.class_count = len(self.train_dataset.class_dict)
        length = len(self.train_dataset)

        trl, tel = int(np.floor(length * 0.8)), int(np.ceil(length * 0.1))
        vel = length - (trl + tel)
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
            (
                self.train_data,
                self.test_data,
            ) = random_split(self.train_dataset, [trl, tel])
            self.val_data = self.test_data
        else:
            self.train_data, self.test_data, self.val_data = random_split(
                self.train_dataset, [trl, tel, vel]
            )
        self.bs = 1

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.bs)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.bs)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.bs)


SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.int()
    labels = labels.int()

    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (
        union + SMOOTH
    )  # We smooth our devision to avoid 0/0
    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds

    return (
        thresholded.mean()
    )  # Or thresholded.mean() if you are interested in average across the batch


class FCNHead_model(pl.LightningModule):
    def __init__(self, classes):
        super(FCNHead_model, self).__init__()
        classes = classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.segmentation.fcn_resnet101(pretrained=True)
        model.classifier = FCNHead(2048, classes)
        model.aux_classifier = FCNHead(1024, classes)
        model.to(device)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, train_batch, batch_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, y = train_batch
        x = x.to(device)
        y = y.to(device)
        logits = self.model(x)
        loss = self.criterion(logits["out"], y.squeeze(0))
        self.log("Training Loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, y = valid_batch
        x = x.to(device)
        y = y.to(device)
        logits = self.model(x)
        loss = self.criterion(logits["out"], y.squeeze(0))
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, y = test_batch
        x = x.to(device)
        y = y.to(device)
        logits = self.model(x)
        loss = self.criterion(logits["out"], y.squeeze(0))
        meaniou = iou_pytorch(logits["out"], y)
        self.log("Testing Loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("Mean IOU:", meaniou)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.params, lr=1e-4)


def train_model(epochs, classes, path, data_loaded):
    max_epoc = 10 if epochs is None else epochs
    model_module = FCNHead_model(classes)
    checkpoint_callback = ModelCheckpoint(monitor="Validation Loss", dirpath=path)

    early_stop_callback = EarlyStopping(
        monitor="Validation Loss", patience=5, verbose=False, mode="min", min_delta=0.05
    )
    devices = torch.cuda.device_count() if torch.cuda.device_count() != 0 else 1
    trainer = pl.Trainer(
        max_epochs=max_epoc,
        devices=devices,
        accelerator="auto",
        benchmark=True,
        default_root_dir=path,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model_module, data_loaded)
    result = trainer.test(datamodule=data_loaded)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def mask_to_rgb(image):
    colour_code = {}  # []
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    classes = np.unique(image).tolist()
    n_classes = len(classes) - 1
    # r[image == 1], g[image == 1], b[image == 1] = [0, 255, 0]
    # r[image == 2], g[image == 2], b[image == 2] = [0, 0, 255]
    # r[image == 3], g[image == 3], b[image == 3] = [255, 0, 0]
    for i in range(1, n_classes + 1):
        random_colour = [
            np.random.randint(255),
            np.random.randint(255),
            np.random.randint(255),
        ]
        r[image == i], g[image == i], b[image == i] = random_colour
        colour_code[f"{i}"] = random_colour
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask, colour_code


def predict_img(
    image_url, model_weight, annotation_file_path, output_save_path, output_file_name
):
    f = requests.get(annotation_file_path)
    data = json.loads(f.text)
    class_count = len(data["categories"]) + 1

    """local path"""
    #     train_dataset = CustomDataset(annotation_file_path)
    #     class_count = len(train_dataset.class_dict)

    model = FCNHead_model.load_from_checkpoint(model_weight, classes=class_count)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    width, height = img.size
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    t_img = transform(img).unsqueeze(0)
    t_img = t_img.to(device)

    model = model.eval()
    out = model(t_img)
    y_pred_final = out["out"]
    pred = y_pred_final.cpu().detach().numpy()
    mask = np.argmax(pred[0], axis=0)
    rgb_mask, colour_code = mask_to_rgb(mask)
    rgb_img = Image.fromarray(rgb_mask)

    final_img = Image.blend(img.resize((224, 224)), rgb_img, alpha=0.3)
    final_img = final_img.resize((width, height))
    final_img.save(output_save_path + output_file_name)
    categorie_dicts = data["categories"]
    categorie_dicts = {str(i["id"]): i["name"] for i in categorie_dicts}

    props = []
    for k, v in colour_code.items():
        d = {}
        try:
            d["label"] = categorie_dicts[str(int(k) - 1)]
        except:
            d["label"] = ""

        try:
            d["color_mask"] = f"rgb({v[0]},{v[1]},{v[2]})"
        except:
            d["color_mask"] = ""
        props.append(d)

    return {
        "image": output_save_path + output_file_name,
        "properties": props,
    }


class Semantic:
    def load_data(self, annotation_json_path, download_path):
        self.download_path = download_path
        r = requests.get(annotation_json_path)

        if os.path.exists(self.download_path):
            open(self.download_path + "segmentation_annotation.json", "wb").write(
                r.content
            )
            print("file downloaded")
        else:
            print("path didn't exist, please check")
        if os.path.isfile(self.download_path + "segmentation_annotation.json"):
            self.json_path = self.download_path + "segmentation_annotation.json"
        else:
            raise Exception("file not there, cant load data")
        # self.json_path = annotation_json_path
        # self.download_path= download_path

        self.data_loaded = DataModulesemantic(self.json_path)
        self.class_count = self.data_loaded.class_count

    def train(self, mode="production"):
        self.epochs = 200 if mode == "production" else 5 if mode == "staging" else 1
        self.model = train_model(
            self.epochs, self.class_count, self.download_path, self.data_loaded
        )
        return self.model
