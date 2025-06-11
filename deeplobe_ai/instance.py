# importing dependent libraries
import os
import gc
import torch
import torch.utils.data
import torchvision
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T

import cv2
import requests
import random

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from PIL import Image


import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import warnings
import json

warnings.filterwarnings("ignore")


def bounding_box(points):
    try:
        x_coordinates, y_coordinates = zip(*points)
    except:
        new_points = [
            [points[0][x], points[0][x + 1]] for x in range(0, len(points[0]) - 1, 2)
        ]
        x_coordinates, y_coordinates = zip(*new_points)

    return [
        int(min(x_coordinates)),
        int(min(y_coordinates)),
        int(max(x_coordinates)),
        int(max(y_coordinates)),
    ]


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform():
    custom_transforms = []
    custom_transforms.append(T.ToTensor())
    return T.Compose(custom_transforms)


class MyOwnDataset(torch.utils.data.Dataset):
    def __init__(self, ann_path):
        self.ann_file = ann_path
        self.coco = COCO(self.ann_file)

        self.catIDs = self.coco.getCatIds()
        self.categories = self.coco.loadCats(self.catIDs)
        self.class_dict = {
            dictionary["id"] + 1: dictionary["name"] for dictionary in self.categories
        }
        self.class_dict[0] = "background"
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_transforms = T.Compose([T.ToTensor()])
        self.mask_transforms = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        image_annotation = self.coco.loadImgs(image_id)[0]

        image_url = self.coco.loadImgs(image_id)[0]["url"]
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        image_annotation["height"] = image.size[1]
        image_annotation["width"] = image.size[0]

        boxes = []
        labels = []
        num_objs = len(coco_annotation)
        for i in range(num_objs):
            box = bounding_box(coco_annotation[i]["segmentation"].copy())
            boxes.append(box)
            labels.append(coco_annotation[i]["category_id"] + 1)

        img_id = torch.tensor([image_id])
        labels = torch.tensor(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # masks loading
        for index, annotation in enumerate(coco_annotation):
            mask = annotation["segmentation"]
            mask = [[point for points in mask for point in points]]
            coco_annotation[index]["segmentation"] = mask

        masks = []
        for ann in coco_annotation:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        masks = np.stack(masks)
        fmask = torch.as_tensor(masks, dtype=torch.uint8)

        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["masks"] = fmask

        image = self.img_transforms(image)
        return image, my_annotation


class DataModuleInstance(pl.LightningDataModule):
    def __init__(
        self,
        json_path,
    ):
        super(DataModuleInstance, self).__init__()
        self.bs = 1
        self.json_path = json_path
        self.dataset = MyOwnDataset(self.json_path)
        self.class_count = self.dataset.class_dict

        length = len(self.dataset)
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
            ) = random_split(self.dataset, [trl, tel])
            self.val_data = self.test_data
        else:
            self.train_data, self.test_data, self.val_data = random_split(
                self.dataset, [trl, tel, vel]
            )
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.bs,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.bs, shuffle=False, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.bs,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


class InstanceSegment(pl.LightningModule):
    def __init__(self, classes):
        super(InstanceSegment, self).__init__()
        self.save_hyperparameters()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.classes = classes
        num_classes = self.classes

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model = model
        self.model.to(device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]

        self.metric = MeanAveragePrecision()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        return torch.optim.SGD(self.params, lr=0.0005, momentum=0.08)

    def training_step(self, train_batch, batch_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        imgs, annotations = train_batch
        imgs = list(img.to(device) for img in imgs)
        annotations = [
            {key: value.to(device) for key, value in annotation.items()}
            for annotation in annotations
        ]

        loss_dict = self.model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses, on_step=True, on_epoch=True, sync_dist=True)
        return losses

    """
    def validation_step(self, valid_batch,batch_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        imgs, annotations  = valid_batch
        imgs = list(img.to(device) for img in imgs)
        annotations1 = [{key: value.to(device) for key, value in annotation.items()} for annotation in annotations]

        preds = self.model(imgs, annotations1)
        loss = self.criterion(preds[0]['boxes'], annotations1[0]['boxes'])
        self.log("val_loss", loss)
        return losss"""

    def test_step(self, test_batch, batch_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        imgs, annotations = test_batch
        imgs = list(img.to(device) for img in imgs)
        annotations1 = [
            {key: value.to(device) for key, value in annotation.items()}
            for annotation in annotations
        ]

        preds = self.model(imgs, annotations1)
        self.metric.update(preds, annotations1)
        result1 = self.metric.compute()
        self.log("MAP Results:", result1)
        return result1


def train_model(epochs, classes, folder_path, data_loaded):
    max_epoch = epochs
    data_module = data_loaded
    model_module = InstanceSegment(classes)

    checkpoint_callback = ModelCheckpoint(monitor="train_loss", dirpath=folder_path)
    early_stop_callback = EarlyStopping(
        monitor="train_loss", patience=5, verbose=False, mode="min", min_delta=0.05
    )
    devices = torch.cuda.device_count() if torch.cuda.device_count() != 0 else 1
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        devices=devices,
        accelerator="auto",
        default_root_dir=folder_path,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(model_module, data_module)
    result = trainer.test(datamodule=data_module)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


class Instance:
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
        self.data_loaded = DataModuleInstance(self.json_path)
        self.class_count = len(self.data_loaded.class_count)

    def train(self, mode="production"):
        self.epochs = 200 if mode == "production" else 5 if mode == "staging" else 1
        self.model = train_model(
            self.epochs, self.class_count, self.download_path, self.data_loaded
        )
        metrics = [
            {key: val.item()} for key, val in self.model[0]["MAP Results:"].items()
        ]
        return metrics


def predict(model, img, threshold, class_dict):
    model = model.eval()
    pred = model([img])

    pred_score = list(pred[0]["scores"].detach().to("cpu").numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    masks = (pred[0]["masks"] > 0.5).squeeze().detach().to("cpu").numpy()
    pred_class = [
        class_dict[i] for i in list(pred[0]["labels"].detach().to("cpu").numpy())
    ]
    pred_boxes = [
        (int(i[0]), int(i[1]), int(i[2]), int(i[3]))
        for i in list(pred[0]["boxes"].detach().to("cpu").numpy())
    ]

    masks = masks[: pred_t + 1]
    boxes = pred_boxes[: pred_t + 1]
    pred_cls = pred_class[: pred_t + 1]
    scores = pred_score[: pred_t + 1]
    return masks, boxes, pred_cls, scores


def random_colour_masks(image):
    colours = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def image_prediction(
    img_file,
    model_weight,
    annotation_file_path,
    output_file_path,
    output_file_name,
    threshold=0.7,
):
    f = requests.get(annotation_file_path)
    data = json.loads(f.text)
    class_dict = {d["id"] + 1: d["name"] for d in data["categories"]}
    class_dict[0] = "background"
    classes = len(class_dict)
    model = InstanceSegment.load_from_checkpoint(
        model_weight, classes=classes
    )  # git suggestion

    image = Image.open(requests.get(img_file, stream=True).raw).convert("RGB")
    width, height = image.size
    transform = T.Compose([T.ToTensor()])
    img = transform(image.copy())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = img.to(device)  # cuda()
    model = model.to(device)
    masks, boxes, pred_cls, scores = predict(model, img, threshold, class_dict)

    np_img = np.array(image)
    color = []
    for i in range(len(masks)):
        color_list = np.random.randint(0, high=256, size=(3,)).tolist()
        rgb_mask = random_colour_masks(masks[i])
        np_img = cv2.addWeighted(np_img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(
            np_img,
            (boxes[i][0], boxes[i][1]),
            (boxes[i][2], boxes[i][3]),
            # color=(0, 255, 0),
            color=color_list,
            thickness=1,
        )

        x_min, y_min = boxes[i][0], boxes[i][1]
        category_name = pred_cls[i]
        color.append(color_list)

        if np_img.shape[0] <= 200 and np_img.shape[1] <= 200:
            fontScale = 2
            thickness = 10
        else:
            fontScale = 0.5
            thickness = 1

        ((text_width, text_height), _) = cv2.getTextSize(
            category_name, cv2.FONT_HERSHEY_DUPLEX, fontScale, thickness
        )
        cv2.rectangle(
            np_img,
            (int(x_min), int(y_min) - int(1.3 * text_height)),
            (int(x_min + text_width), int(y_min)),
            (0, 240, 0),
            -1,
        )

        np_img = cv2.putText(
            np_img,
            category_name,
            (int(x_min), int(y_min) - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=fontScale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    np_img = cv2.resize(np_img, (width, height))
    filename = output_file_path + output_file_name
    cv2.imwrite(filename, np_img)
    score = [i.item() for i in scores]
    props = []
    for i in range(len(boxes)):
        d = {}
        try:
            d["label"] = pred_cls[i]
        except:
            d["label"] = ""

        try:
            d["predicted_score"] = score[i]
        except:
            d["predicted_score"] = ""

        try:
            d["coordinates"] = [
                {"x1": boxes[i][0], "y1": boxes[i][1] + boxes[i][3]},
                {"x2": boxes[i][0] + boxes[i][2], "y2": boxes[i][1] + boxes[i][3]},
                {"x3": boxes[i][0] + boxes[i][2], "y3": boxes[i][1]},
                {"x4": boxes[i][0], "y4": boxes[i][1]},
            ]
        except:
            d["coordinates"] = ""

        try:
            d["color"] = f"rgb({color[i][0]},{color[i][1]},{color[i][2]})"
        except:
            d["color"] = ""
        props.append(d)

    return {"image": filename, "properties": props}

    # {
    #     "image": filename,
    #     "labels": pred_cls,
    #     "scores": score,
    #     "bboxes": boxes,
    #     "colours": color,
    # }
    # palt.show()


# def image_prediction(
#     img_file,
#     model_weight,
#     annotation_file_path,
#     threshold,
#     output_file_path,
#     output_file_name,
# ):
#     f = requests.get(annotation_file_path)
#     data = json.loads(f.text)
#     class_dict = {d["id"] + 1: d["name"] for d in data["categories"]}
#     class_dict[0] = "background"
#     classes = len(class_dict)

#     # model=InstanceSegment(classes)
#     # model.load_from_checkpoint(model_weight)
#     model = InstanceSegment.load_from_checkpoint(
#         model_weight, classes=classes
#     )  # git suggestion

#     image = Image.open(requests.get(img_file, stream=True).raw).convert("RGB")
#     transform = T.Compose([T.ToTensor()])
#     img = transform(image.copy())
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     img = img.to(device)  # cuda()
#     model = model.to(device)

#     masks, boxes, pred_cls = predict(model, img, threshold, class_dict)

#     # print(len(masks))
#     np_img = np.array(image)
#     for i in range(len(masks)):
#         # plt.imshow(masks[i])
#         # plt.show()
#         rgb_mask = random_colour_masks(masks[i])
#         np_img = cv2.addWeighted(np_img, 1, rgb_mask, 0.5, 0)
#         cv2.rectangle(
#             np_img,
#             (boxes[i][0], boxes[i][1]),
#             (boxes[i][2], boxes[i][3]),
#             color=(0, 255, 0),
#             thickness=1,
#         )
#         cv2.putText(
#             np_img,
#             pred_cls[i],
#             (boxes[i][0], boxes[i][1]),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             3,
#             (0, 255, 0),
#             3,
#         )
#     filename = output_file_path + output_file_name
#     cv2.imwrite(filename, np_img)
#     fig, axs = plt.subplots(1, 1, figsize=(20, 20))
#     axs.imshow(np_img)
#     return filename
#     # palt.show()


# json_path = 'https://deeplobe.s3.amazonaws.com/instance/693bc4d4-9bd9-4afa-a079-c89ebb516ec6/Annotate_iF2pDPq.json'
# download_path = '/home/ubuntu/soulpage/Bhavana/Final_Codes/instance_model/'
# download_path = '/home/ubuntu/soulpage/Bhavana/Final_Codes/'
# inst = Instance()
# inst.load_data(json_path, download_path)
# inst.train(demo=False)

# img_url = "https://deeplobe.s3.amazonaws.com/media_assets/00002_N7MySf5_narbrCu.jpg"
# img_url = "https://deeplobe.s3.amazonaws.com/media_assets/00007_IDHoHhd_xY7ifJG.jpg"
# save_path = '/home/ubuntu/soulpage/Bhavana/Final_Codes/result_imgs/'
# save_name = 'instance_output2.jpg'
# model_weight = '/home/ubuntu/soulpage/Bhavana/Final_Codes/instance_model/epoch=12-step=415.ckpt'
# output = image_prediction(img_url, model_weight, json_path,threshold= 0.3, output_file_path= save_path, output_file_name= save_name)
# # print(output)
