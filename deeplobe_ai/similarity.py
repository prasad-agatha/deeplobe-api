import random
import gc

import numpy as np

import torch
import requests
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


import warnings

warnings.filterwarnings("ignore")
from PIL import Image
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from deeplobe_ai.preprocess import *


def process_data(total_imgs):
    train = []
    test = []
    valid = []
    class2index = {
        key: i for i, key in enumerate((total_imgs.keys()))
    }  # total_imgs is a dict

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


class Sim_dataset(torch.utils.data.Dataset):
    def __init__(
        self, path, trans, auto_orient_bool, custom_resize_bool, output_size, fit_method
    ):
        self.data_path = path
        self.transform = trans
        self.images = [img_row[0] for img_row in self.data_path]
        self.labels = [img_row[1] for img_row in self.data_path]
        self.triplets = []

        self.auto_orient_bool = auto_orient_bool
        self.custom_resize_bool = custom_resize_bool
        self.output_size = output_size
        self.fit_method = fit_method

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgurl = self.images[idx]
        # anchor_image = self.transform(Image.open(requests.get(imgurl, stream=True).raw).convert("RGB")  # anchor image)

        image = Image.open(requests.get(imgurl, stream=True).raw).convert("RGB")

        if self.auto_orient_bool == True:
            # auto-orient image based on exif data
            image = auto_orient(image)
        if self.custom_resize_bool == True:
            # resize image based on user input
            image = resize_image(image, self.output_size, self.fit_method)

        anchor_image = self.transform(image)  # anchor image

        tar_label = self.labels[idx]

        pos_indexes = [ind for ind, i in enumerate(self.labels) if i == tar_label]
        neg_indexes = [ind for ind, i in enumerate(self.labels) if i != tar_label]

        pos_index_img_url = self.images[random.choice(pos_indexes)]
        pos_index_img = Image.open(
            requests.get(pos_index_img_url, stream=True).raw
        ).convert("RGB")

        if self.auto_orient_bool == True:
            # auto-orient image based on exif data
            pos_index_img = auto_orient(pos_index_img)
        if self.custom_resize_bool == True:
            # resize image based on user input
            pos_index_img = resize_image(
                pos_index_img, self.output_size, self.fit_method
            )

        pos_image = self.transform(pos_index_img)  # positive image

        neg_index_img_url = self.images[random.choice(neg_indexes)]
        # neg_image = self.transform(Image.open(requests.get(neg_index_img_url, stream=True).raw).convert("RGB"))  # negative image
        neg_index_img = Image.open(
            requests.get(neg_index_img_url, stream=True).raw
        ).convert("RGB")

        if self.auto_orient_bool == True:
            # auto-orient image based on exif data
            neg_index_img = auto_orient(neg_index_img)
        if self.custom_resize_bool == True:
            # resize image based on user input
            neg_index_img = resize_image(
                neg_index_img, self.output_size, self.fit_method
            )
        neg_image = self.transform(neg_index_img)  # negative image

        self.triplets.append(
            "Anchor:"
            + self.images[idx]
            + " pos_img "
            + self.images[random.choice(pos_indexes)]
            + " neg_img "
            + self.images[random.choice(neg_indexes)]
        )
        return anchor_image, pos_image, neg_image


class Dataset(pl.LightningDataModule):
    def __init__(
        self, all_images, auto_orient_bool, custom_resize_bool, output_size, fit_method
    ):
        super().__init__()

        self.all_images = all_images
        self.batch_size = 4
        self.auto_orient_bool = auto_orient_bool
        self.custom_resize_bool = custom_resize_bool
        self.output_size = output_size
        self.fit_method = fit_method

        transform_list = []
        # Add default resize and auto-orient if auto-preprocessing is selected
        if custom_resize_bool == False:
            transform_list.append(transforms.Resize((224, 224)))

        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform = transforms.Compose(transform_list)
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.trainset, self.testset, self.validset, self.class2index = process_data(
            self.all_images
        )

    def setup(self, stage="fit"):
        self.train_set = Sim_dataset(
            self.trainset,
            self.transform,
            self.auto_orient_bool,
            self.custom_resize_bool,
            self.output_size,
            self.fit_method,
        )
        self.test_set = Sim_dataset(
            self.testset,
            self.transform,
            self.auto_orient_bool,
            self.custom_resize_bool,
            self.output_size,
            self.fit_method,
        )
        self.val_set = Sim_dataset(
            self.validset,
            self.transform,
            self.auto_orient_bool,
            self.custom_resize_bool,
            self.output_size,
            self.fit_method,
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.basemodel = self.model.features
        self.conv1 = nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=(7, 7))
        self.batch1 = nn.BatchNorm2d(1024)
        self.drop1 = nn.Dropout2d(0.45)

        self.fc1 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.basemodel(x)
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = self.drop1(x)

        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class Similarity_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Net()
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, x):
        self.net = self.net.eval()
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x1, x2, x3 = batch
        output1, output2, output3 = self.net(x1), self.net(x2), self.net(x3)
        loss = self.criterion(output1, output2, output3)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x1, x2, x3 = batch
        output1, output2, output3 = self.net(x1), self.net(x2), self.net(x3)
        loss = self.criterion(output1, output2, output3)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x1, x2, x3 = batch
        output1, output2, output3 = self.net(x1), self.net(x2), self.net(x3)
        loss = self.criterion(output1, output2, output3)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def cosine_distance(input1, input2):
    # return np.dot(input1, input2) / (np.linalg.norm(input1) * np.linalg.norm(input2))
    return np.round(
        np.dot(input1, input2.T)
        / np.dot(
            np.linalg.norm(input1, axis=1, keepdims=True),
            np.linalg.norm(input2.T, axis=0, keepdims=True),
        ),
        2,
    )


class Imagesimilarity:
    def load_data(
        self,
        url_dict,
        save_model_path,
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
        self.total_imgs = total_imgs
        self.auto_orient_bool = auto_orient_bool
        self.custom_resize_bool = custom_resize_bool
        self.output_size = output_size
        self.fit_method = fit_method
        self.data1 = Dataset(
            self.total_imgs,
            self.auto_orient_bool,
            self.custom_resize_bool,
            self.output_size,
            self.fit_method,
        )
        self.save_model_path = save_model_path
        # self.data1.setup()

    def train(self, mode="production"):
        epochs = 200 if mode == "production" else 5 if mode == "staging" else 1
        model = Similarity_model()
        print("Model Training started")
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=self.save_model_path
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.05, patience=5, verbose=False, mode="min"
        )
        devices = torch.cuda.device_count() if torch.cuda.device_count() != 0 else 1
        trainer = pl.Trainer(
            devices=devices,
            accelerator="auto",
            max_epochs=epochs,
            default_root_dir=self.save_model_path,
            callbacks=[self.checkpoint_callback, early_stop_callback],
        )

        trainer.fit(model, self.data1)
        print("Testing the model now")
        result = trainer.test(datamodule=self.data1)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return result


def predict(selected_model, img2_url_list, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}
    imag2 = []
    ss = []
    for i in range(len(img2_url_list)):
        try:
            img1 = Image.open(requests.get(img2_url_list[0], stream=True).raw).convert(
                "RGB"
            )

            img2 = Image.open(
                requests.get(img2_url_list[i + 1], stream=True).raw
            ).convert("RGB")

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            t_img1, t_img2 = transform(img1), transform(img2)
            model = Similarity_model.load_from_checkpoint(selected_model)
            model = model.eval().to(device)
            out1, out2 = model(t_img1.unsqueeze(dim=0).to(device)), model(
                t_img2.unsqueeze(dim=0).to(device)
            )
            result = cosine_distance(
                out1.detach().cpu().numpy(), out2.detach().cpu().numpy()
            )
            print(
                "images are similar with score " + str(result[0][0])
                if result[0][0] > thresh
                else "images are not similar with score " + str(result[0][0])
            )

            imag1 = img2_url_list[0]
            imag2.append(img2_url_list[i + 1])
            ss.append(str(result[0][0]))
        except:
            print("no more images")

    results["image1"] = imag1
    results["image2"] = imag2
    results["similarity_score"] = ss
    return results
