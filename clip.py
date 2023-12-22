import gc
import numpy as np
import pandas as pd
import os
import cv2
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class CLIPNetwork(nn.Module):
    def __init__(
        self,
        temp_factor=CFG.temperature,
        img_embedding_size=CFG.image_embedding,
        txt_embedding_size=CFG.text_embedding,
    ):
        super(CLIPNetwork, self).__init__()
        self.visual_encoder = ImageEncoder()
        self.language_decoder = Decoder()
        self.visual_projection_head = ProjectionHead(embedding_dim=img_embedding_size)
        self.language_projection_head = ProjectionHead(embedding_dim=txt_embedding_size)
        self.temp_factor = temp_factor

    def forward(self, input_batch):
        # Extract features from images and texts
        visual_features = self.visual_encoder(input_batch["image"])
        language_features = self.language_decoder(
            input_ids=input_batch["input_ids"], attention_mask=input_batch["attention_mask"]
        )
        # Project features to embeddings with the same dimensions
        visual_embeddings = self.visual_projection_head(visual_features)
        language_embeddings = self.language_projection_head(language_features)

        # Loss calculation
        sim_logits = (language_embeddings @ visual_embeddings.T) / self.temp_factor
        visual_self_similarity = visual_embeddings @ visual_embeddings.T
        language_self_similarity = language_embeddings @ language_embeddings.T
        similarity_targets = F.softmax(
            (visual_self_similarity + language_self_similarity) / 2 * self.temp_factor, dim=-1
        )
        language_loss = cross_entropy_loss(sim_logits, similarity_targets, reduction_type='none')
        visual_loss = cross_entropy_loss(sim_logits.T, similarity_targets.T, reduction_type='none')
        total_loss = (visual_loss + language_loss) / 2.0  # shape: (batch_size)
        return total_loss.mean()


def cross_entropy_loss(predictions, target_values, reduction_type='none'):
    softmax_log = nn.LogSoftmax(dim=-1)
    computed_loss = (-target_values * softmax_log(predictions)).sum(1)
    if reduction_type == "none":
        return computed_loss
    elif reduction_type == "mean":
        return computed_loss.mean()