import torch

class CFG:
    debug = False
    batch_size = 32
    num_workers = 2
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = image_path
    captions_path = captions_path

    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5

    model_name = 'resnet50'
    text_encoder_model = "distilbert-base-uncased"
    text_tokenizer = "distilbert-base-uncased"

    image_embedding = 2048
    text_embedding = 768
    num_projection_layers = 1
    projection_dim = 256

    weight_decay = 1e-3
    patience = 1
    factor = 0.7
    temperature = 1.0
    max_length = 200

    pretrained = True
    trainable = True

    size = 380
    dropout = 0.1
