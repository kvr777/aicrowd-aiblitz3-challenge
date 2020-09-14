import os
import random

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomAffine,
    ToPILImage,
)
from skimage import io
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import MNIST
from PIL import Image
from simplenet import simplenet


IMG_PATH = "data/train_classification"
LABEL_PATH = "data/train_classification_label.csv"
CHECKPOINT_DIR = "models"
SEED = 42
N_FOLDS = 10
USE_FOLD = 0
VAL_TEST_RATIO = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


class SudokuDataset(Dataset):
    def __init__(self, fold, split, transform):
        self.split = split
        self.fold = fold
        self.transform = transform

    def __len__(self):
        return len(SPLIT_DICT[self.fold][self.split][0])

    def __getitem__(self, idx):
        # print("idx: ", idx)
        img_name_prefix = SPLIT_DICT[self.fold][self.split][0][idx]
        img_path = os.path.join(IMG_PATH, f"{img_name_prefix}.png")

        if len(SPLIT_DICT[self.fold][self.split]) == 2:
            img_label = SPLIT_DICT[self.fold][self.split][1][idx]
        else:
            img_label = None

        # print("img path", img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(img_label - 1, dtype=torch.long)


class SudokuModel(pl.LightningModule):
    def __init__(self):
        super(SudokuModel, self).__init__()
        self.model = simplenet(classes=9)

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        # print("SSSSSSSSSSSSSSSSSSSSss", y_hat.shape, y.shape)
        return {"val_loss": F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {"test_loss": F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            SudokuDataset(
                fold=USE_FOLD, split="train", transform=TRANSFORM_DICT["train"]
            ),
            batch_size=128,
            drop_last=True,
            num_workers=4,
        )

        # fold, split, transform

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            SudokuDataset(
                fold=USE_FOLD, split="valid", transform=TRANSFORM_DICT["valid"]
            ),
            batch_size=200,
            drop_last=True,
            num_workers=4,
        )

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            SudokuDataset(
                fold=USE_FOLD, split="test", transform=TRANSFORM_DICT["test"]
            ),
            batch_size=128,
            drop_last=True,
            num_workers=4,
        )


if __name__ == "__main__":
    labels_df = pd.read_csv(LABEL_PATH)
    imgs, labels = labels_df["id"].values, labels_df["label"].values
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED)
    SPLIT_DICT = dict()
    fold = 0
    for train_index, valid_test_index in skf.split(imgs, labels):
        val_test_break_point = int(VAL_TEST_RATIO * len(valid_test_index))
        valid_index, test_index = (
            valid_test_index[:val_test_break_point],
            valid_test_index[val_test_break_point:],
        )
        X_train, X_valid, X_test = (
            imgs[train_index],
            imgs[valid_index],
            imgs[test_index],
        )
        y_train, y_valid, y_test = (
            labels[train_index],
            labels[valid_index],
            labels[test_index],
        )
        SPLIT_DICT[fold] = {
            "train": [X_train, y_train],
            "valid": [X_valid, y_valid],
            "test": [X_test, y_test],
        }
        fold += 1

    TRANSFORM_DICT = {
        "train": Compose(
            [
                ToPILImage(),
                RandomAffine(degrees=(-10, 10), scale=(0.9, 1.1), translate=(0.1, 0.1)),
                ToTensor(),
                Normalize((0.5,), (1.0,)),
            ]
        ),
        "valid": Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
        "test": Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    }

    for split in ["train", "valid", "test"]:
        print(split, len(SPLIT_DICT[0][split][0]))

    sudoku_model = SudokuModel()

    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_DIR,
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    trainer = pl.Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        log_save_interval=100,
    )
    trainer.fit(sudoku_model)
    trainer.test()
