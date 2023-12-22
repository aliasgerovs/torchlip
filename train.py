import gc
import numpy as np
import pandas as pd
import os
import cv2
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformation import *
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from utils import CFG

def split_dataset_into_train_and_valid():
    caption_data = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_image_id = caption_data["id"].max() + 1 if not CFG.debug else 100
    all_image_ids = np.arange(0, max_image_id)
    np.random.seed(42)
    validation_image_ids = np.random.choice(
        all_image_ids, size=int(0.2 * len(all_image_ids)), replace=False
    )
    training_image_ids = [image_id for image_id in all_image_ids if image_id not in validation_image_ids]
    training_set = caption_data[caption_data["id"].isin(training_image_ids)].reset_index(drop=True)
    validation_set = caption_data[caption_data["id"].isin(validation_image_ids)].reset_index(drop=True)
    return training_set, validation_set


def create_data_loader(data, text_tokenizer, current_mode):
    image_transformations = get_transforms(mode=current_mode)
    clip_dataset = Dataset(
        images=data["image"].values,
        captions=data["caption"].values,
        tokenizer=text_tokenizer,
        transforms=image_transformations,
    )
    loader = torch.utils.data.DataLoader(
        clip_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=current_mode == "train",
    )
    return loader
