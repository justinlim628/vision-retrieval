import torch
import open_clip
import numpy as np


def load_model(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.config import CLIP_MODEL, CLIP_PRETRAINED
    model, preprocess, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model.eval()
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer, device


def encode_image(image, model, preprocess, device) -> np.ndarray:
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()


def encode_text(query: str, model, tokenizer, device) -> np.ndarray:
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()
