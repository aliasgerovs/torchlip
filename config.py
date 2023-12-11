import gc
import numpy as np
import pandas as pd
import os
class CFG:
    # Basic configuration
    debug = False
    batch_size = 32
    num_workers = 2
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    image_path = image_path
    captions_path = captions_path

    # Learning rates
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5

    # Model configuration
    model_name = 'resnet50'
    text_encoder_model = "distilbert-base-uncased"
    text_tokenizer = "distilbert-base-uncased"

    # Embedding and projection dimensions
    image_embedding = 2048
    text_embedding = 768
    num_projection_layers = 1
    projection_dim = 256

    # Training parameters
    weight_decay = 1e-3
    patience = 1
    factor = 0.7
    temperature = 1.0
    max_length = 200

    # Pretraining and trainability
    pretrained = True
    trainable = True

    # Image processing
    size = 380  # image size
    dropout = 0.1