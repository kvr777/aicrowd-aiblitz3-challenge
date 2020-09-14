import os
import random
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from train_classifier import SudokuDataset, SudokuModel
from torch.nn import functional as F
import operator

from torchvision.transforms import Compose, ToTensor, Normalize


IMG_DIR = "data/test/images"
CHECKPOINT_DIR = "models"
SEED = 42
N_FOLDS = 10
USE_FOLD = 0
VAL_TEST_RATIO = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


transform = Compose([ToTensor(), Normalize((0.5,), (1.0,))])


def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def rotate_and_resize(img):
    contours, h = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    top_right, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )

    crop_rect = [
        polygon[top_left][0],
        polygon[top_right][0],
        polygon[bottom_right][0],
        polygon[bottom_left][0],
    ]

    top_left, top_right, bottom_right, bottom_left = (
        crop_rect[0],
        crop_rect[1],
        crop_rect[2],
        crop_rect[3],
    )
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    side = max(
        [
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right),
        ]
    )

    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32"
    )
    m = cv2.getPerspectiveTransform(src, dst)
    cropped = cv2.warpPerspective(img, m, (int(side), int(side)))
    resized = cv2.resize(cropped, (306, 306))
    proc = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    proc = cv2.bitwise_not(proc, proc)
    return proc


def recognize_numbers_from_img(img_name, dct):
    img = cv2.imread(os.path.join(IMG_DIR, img_name), cv2.IMREAD_GRAYSCALE)
    img = rotate_and_resize(img)
    img_name = img_name.split(".")[0]
    mnist_size = 28
    center_crop_pct = 0.5
    thresh = 500
    dst = np.zeros_like(img)
    h, w = img.shape
    assert h == w
    step = int(h / 9)
    pad = step - mnist_size
    assert pad % 2 == 0
    pad = int(pad / 2)
    # fig=plt.figure(figsize=(8, 8))
    cnt = 1
    for i in range(9):
        for j in range(9):
            sq_h_s, sq_h_e = i * step, (i + 1) * step
            sq_w_s, sq_w_e = j * step, (j + 1) * step
            # print(sq_h_s, sq_h_e, sq_w_s, sq_w_e)
            cur_square = img[sq_h_s + pad : sq_h_e - pad, sq_w_s + pad : sq_w_e - pad]
            sq_h, sq_w = cur_square.shape
            # print(sq_h, sq_w)
            crop_h, crop_w = (
                int((center_crop_pct * sq_h) / 2),
                int((center_crop_pct * sq_w) / 2),
            )
            cur_square_croped = cur_square[
                crop_h : sq_h - crop_h, crop_w : sq_w - crop_w
            ]
            # fig.add_subplot(9, 9, cnt)
            # plt.imshow(cur_square_croped)
            cnt += 1
            cur_sum = sum(sum(cur_square_croped))
            if cur_sum >= thresh:
                label_id = f"{img_name}_{i}_{j}"
                digit_img = img[sq_h_s:sq_h_e, sq_w_s:sq_w_e]

                digit_img = transform(digit_img)
                digit_img = digit_img.unsqueeze(0).cuda()

                preds = F.softmax(model(digit_img), dim=1)

                pred = preds.argmax(-1).detach().cpu().numpy()[0]

                dct["id"].append(label_id)
                dct["preds"].append(pred + 1)


if __name__ == "__main__":

    last_file = sorted(os.listdir(CHECKPOINT_DIR))[-1]
    model = SudokuModel.load_from_checkpoint(
        checkpoint_path=os.path.join(CHECKPOINT_DIR, last_file)
    )
    model.eval()
    model.cuda()

    img_files = os.listdir(IMG_DIR)

    dct = dict()
    dct["id"] = []
    dct["preds"] = []

    for fname in tqdm(img_files):
        recognize_numbers_from_img(fname, dct)

    res_pd = pd.DataFrame.from_dict(dct)

    res_pd.to_csv("data/test_classification_preds.csv", index=False)
