import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DATA_DIR = Path("data/coco")
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.csv"
IMAGES_DIR = DATA_DIR / "images" / "val2017"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

DEFAULT_K = 9
LABEL_K = 10
CATEGORIES = ['person', 'car', 'truck', 'dog', 'cat']
