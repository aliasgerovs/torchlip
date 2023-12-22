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
from utils import Average
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

def train_one_epoch(model, data_loader, optim, scheduler, lr_update_mode):
    avg_loss = Average()
    progress = tqdm(data_loader, total=len(data_loader))
    for data in progress:
        data = {key: val.to(CFG.device) for key, val in data.items() if key != "caption"}
        loss_val = model(data)
        optim.zero_grad()
        loss_val.backward()
        optim.step()
        if lr_update_mode == "batch":
            scheduler.step()

        batch_size = data["image"].size(0)
        avg_loss.update(loss_val.item(), batch_size)

        progress.set_postfix(train_loss=avg_loss.avg, lr=current_lr(optim))
    return avg_loss


def validate_model(model, data_loader):
    avg_loss = Average()

    progress = tqdm(data_loader, total=len(data_loader))
    for data in progress:
        data = {key: val.to(CFG.device) for key, val in data.items() if key != "caption"}
        loss_val = model(data)

        batch_size = data["image"].size(0)
        avg_loss.update(loss_val.item(), batch_size)

        progress.set_postfix(valid_loss=avg_loss.avg)
    return avg_loss


def execute_training():
    training_set, validation_set = prepare_train_valid_data()
    text_tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    training_loader = create_loaders(training_set, text_tokenizer, "train")
    validation_loader = create_loaders(validation_set, text_tokenizer, "valid")

    clip_network = CLIPModel().to(CFG.device)
    model_params = [
        {"params": clip_network.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": clip_network.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            clip_network.image_projection.parameters(), clip_network.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optim = torch.optim.AdamW(model_params, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    lr_update_mode = "epoch"

    record_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        clip_network.train()
        training_loss = train_one_epoch(clip_network, training_loader, optim, scheduler, lr_update_mode)
        clip_network.eval()
        with torch.no_grad():
            validation_loss = validate_model(clip_network, validation_loader)

        if validation_loss.avg < record_loss:
            record_loss = validation_loss.avg
            torch.save(clip_network.state_dict(), "best_model.pt")
            print("Saved Best Model!")

        scheduler.step(validation_loss.avg)
