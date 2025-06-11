import gc
import numpy as np

import torch
import requests
import torch.nn as nn

from torchvision import transforms, models
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
from deeplobe_ai.preprocess import *


def process_data(total_imgs):
    train = []
    test = []
    valid = []
    class2index = {key: i for i, key in enumerate((total_imgs.keys()))}

    for category in total_imgs.keys():
        catg_length = len(total_imgs[category])
        if catg_length <= 10:
            print("test and valid are same due to low data")
            trainl = total_imgs[category][: int(np.floor(0.8 * catg_length))]
            testl = total_imgs[category][int(np.floor(0.8 * catg_length)) :]
            validl = testl
            print(category, "total :", len(total_imgs[category]))
            print("train and test :", len(trainl), len(testl), len(validl))
        else:
            trainl = total_imgs[category][0 : int(np.floor(0.8 * catg_length))]
            others = total_imgs[category][int(np.floor(0.8 * catg_length)) :]
            testl = others[: int(np.ceil(len(others) * 0.5))]
            validl = total_imgs[category][len(trainl) + len(testl) :]

            print(category, "total :", len(total_imgs[category]))
            print("train and test :", len(trainl), len(testl), len(validl))

        train.extend(trainl)
        test.extend(testl)
        valid.extend(validl)
    return train, test, valid, class2index


class Datasetclass:
    def __init__(
        self,
        data,
        class2index,
        auto_orient_bool,
        custom_resize_bool,
        output_size,
        fit_method,
        transform=None,
    ):
        if transform is None:
            transform_list = []
            # Add default resize if auto-preprocessing is selected
            if custom_resize_bool == False:
                transform_list.append(transforms.Resize((256, 256)))

            transform_list.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.transform = transforms.Compose(transform_list)

        self.data = data
        self.class2index = class2index
        self.auto_orient_bool = auto_orient_bool
        self.custom_resize_bool = custom_resize_bool
        self.output_size = output_size
        self.fit_method = fit_method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        imgurl = self.data[index]
        image = Image.open(requests.get(imgurl[0], stream=True).raw).convert("RGB")

        if self.auto_orient_bool == True:
            # auto-orient image based on exif data
            image = auto_orient(image)
        if self.custom_resize_bool == True:
            # resize image based on user input
            image = resize_image(image, self.output_size, self.fit_method)

        img = self.transform(image)

        label = imgurl[1]
        labelid = self.class2index[label]
        return img, torch.tensor(labelid)


class loading_data(pl.LightningDataModule):
    """A Pytorch lightning data class that
    reads the directory, implements transformations,
    loads into a dataloader and returns the same"""

    def __init__(
        self, all_images, auto_orient_bool, custom_resize_bool, output_size, fit_method
    ):
        super().__init__()

        self.all_images = all_images
        self.batch_size = 16
        self.auto_orient_bool = auto_orient_bool
        self.custom_resize_bool = custom_resize_bool
        self.output_size = output_size
        self.fit_method = fit_method
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.trainset, self.testset, self.validset, self.class2index = process_data(
            self.all_images
        )

    def setup(self, stage="fit"):
        self.train_set = Datasetclass(
            self.trainset,
            self.class2index,
            self.auto_orient_bool,
            self.custom_resize_bool,
            self.output_size,
            self.fit_method,
        )
        self.test_set = Datasetclass(
            self.testset,
            self.class2index,
            self.auto_orient_bool,
            self.custom_resize_bool,
            self.output_size,
            self.fit_method,
        )
        self.val_set = Datasetclass(
            self.validset,
            self.class2index,
            self.auto_orient_bool,
            self.custom_resize_bool,
            self.output_size,
            self.fit_method,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)


def build_model(model, modelname, classes):
    """returns the model based on the model name along
    with number of classes of dataset as a final layer
    in model"""
    if modelname in ["resnet", "inception", "googlenet"]:
        infeatures = model.fc.in_features
        model.fc = nn.Linear(infeatures, classes)
    if modelname in ["vggnet", "alexnet"]:
        infeatures = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(infeatures, classes)
    if modelname in ["mobilenet"]:
        infeatures = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(infeatures, classes)
    if modelname in ["densenet"]:
        infeatures = model.classifier.in_features
        model.classifier = nn.Linear(infeatures, classes)
    return model


class Classificationmodel(pl.LightningModule):
    """Core model where it takes resnet18 by default
    creates a training step, validation step and
    test step. it also has configure optimizers"""

    def __init__(self, classes):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(pretrained=True)
        self.classes = classes
        self.model = build_model(self.model, "resnet", self.classes)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 0.001
        self.accuracy = Accuracy(task="multiclass", num_classes=self.classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        # loss = self.criterion(output, y)
        acc = self.accuracy(output, y)
        if torch.cuda.is_available() == True:
            acc = acc.cpu().item()
        # self.log("performance",{"test_loss":loss,"Accuracy":acc})
        self.log("performance", {"Accuracy": acc})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


class Classification:
    def load_data(
        self,
        url_dict,
        save_model_loc,
        auto_orient_bool,
        custom_resize_bool,
        output_size,
        fit_method,
    ):
        total_imgs = {}
        for i in url_dict:
            if i.class_name not in total_imgs.keys():
                total_imgs[i.class_name] = []
                total_imgs[i.class_name].append([i.asset.url, i.class_name])
            else:
                total_imgs[i.class_name].append([i.asset.url, i.class_name])

        self.output_size = output_size
        self.fit_method = fit_method
        self.auto_orient_bool = auto_orient_bool
        self.custom_resize_bool = custom_resize_bool
        self.dataload = loading_data(
            total_imgs, auto_orient_bool, custom_resize_bool, output_size, fit_method
        )
        self.class2index = self.dataload.class2index
        self.classcount = len(self.class2index)
        self.save_model_path = save_model_loc

    def train(self, mode="production"):
        model = Classificationmodel(self.classcount)
        print("model training started")
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=self.save_model_path
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=5, verbose=False, mode="min", min_delta=0.05
        )
        devices = torch.cuda.device_count() if torch.cuda.device_count() != 0 else 1
        epochs = 200 if mode == "production" else 5 if mode == "staging" else 1
        trainer = pl.Trainer(
            devices=devices,
            accelerator="auto",
            max_epochs=epochs,
            benchmark=True,
            default_root_dir=self.save_model_path,
            callbacks=[self.checkpoint_callback, early_stop_callback],
        )

        trainer.fit(model, self.dataload)
        print("testing the model now")
        result = trainer.test(datamodule=self.dataload)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return result[0]["performance"]["Accuracy"].item()


def predict(selected_model, url_dict, img):
    img = Image.open(requests.get(img, stream=True).raw).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(([0.485, 0.456, 0.406]), ([0.229, 0.224, 0.225])),
        ]
    )
    t_img = transform(img).unsqueeze(0)
    model = Classificationmodel.load_from_checkpoint(selected_model)
    model = model.eval()
    out = model(t_img)
    softmax = torch.nn.Softmax()
    probs = softmax(out)
    _, pred = out.max(1)
    class_prob = probs[0][pred].item() * 100

    class_dict = {}
    for cls in url_dict:
        if cls.class_name not in class_dict.keys():
            class_dict[cls.class_name] = []
    classes = sorted([cls for cls, val in class_dict.items()])
    final = {cls: pbs.item() for cls, pbs in (zip(classes, list(probs[0])))}
    final = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return final