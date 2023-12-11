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
from tqdm.autonotebook import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


df = pd.read_csv("captions.txt")
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
df.to_csv("captions.csv", index=False)
df = pd.read_csv("captions.csv")
image_path = "/content/Images"
captions_path = "/content"