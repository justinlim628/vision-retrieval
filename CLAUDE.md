# vision_retrieval

A CLIP-based semantic image search and auto-labeling pipeline over the COCO val2017 dataset, with a Gradio web UI for interactive text-to-image retrieval.

## Project Structure

```
vision_retrieval/
├── src/
│   ├── config.py            # Paths and constants (single source of truth)
│   ├── clip_model.py        # CLIP model loading and text encoding
│   └── search.py            # FAISS index loading and similarity search
├── app.py                   # Gradio web app entrypoint
├── notebooks/
│   ├── 00_coco_dataset_exp.ipynb   # COCO annotation parsing → metadata.csv
│   └── 01_clip_retrieval.ipynb     # Full pipeline exploration (embeddings, search, labeling, curation)
├── data/coco/
│   ├── images/val2017/      # 3141 COCO val2017 images (5 categories)
│   ├── annotations/         # COCO annotation JSON files
│   ├── embeddings.npy       # Precomputed CLIP embeddings (3141 × 512 float32)
│   ├── embedding_indices.npy
│   ├── faiss.index          # FAISS IndexFlatIP for cosine similarity search
│   └── metadata.csv         # Image metadata: id, file_name, file_path, labels
├── CLAUDE.md
└── requirements.txt
```

## Tech Stack

| Component | Library |
|---|---|
| Vision-language model | `open_clip_torch` — ViT-B-32, pretrained on OpenAI |
| Vector search | `faiss-cpu` — IndexFlatIP (cosine similarity over L2-normalized vectors) |
| Deep learning | `torch` (CPU only) |
| Dataset | COCO val2017, filtered to 5 categories: person, car, truck, dog, cat |
| Web UI | `gradio` |
| Data handling | `pandas`, `Pillow` |

## Execution

All commands must be run from the project root (`vision_retrieval/`).

**Activate environment:**
```bash
conda activate vision-retrieval
```

**Run the Gradio app:**
```bash
python app.py
```
Opens at `http://127.0.0.1:7860`. The app loads the CLIP model and FAISS index once at startup — first launch takes ~30 seconds.

**Install dependencies (first time):**
```bash
pip install -r requirements.txt
```

## Data Flow (search feature)

```
User text query
  → encode_text()        # CLIP tokenize + encode + L2 normalize → (1, 512)
  → faiss.search()       # cosine similarity over 3141 embeddings
  → metadata.iloc[...]   # map FAISS positions → image rows
  → PIL.Image.open()     # load from data/coco/images/val2017/
  → gr.Gallery           # display with similarity score captions
```

## Code Style

- Language: Python
- Comments and string literals: English, single quotes
- Notebooks for prototyping only, reusable logic goes in `src/`
- All paths relative to project root

## Development Notes

- Notebooks are for exploration only — do not add new logic there
- All reusable logic goes in `src/`
- Data paths: always relative to project root (`vision_retrieval/`)

## Known Issues & Fixes

### Image paths in metadata.csv are relative to notebooks/
`data/coco/metadata.csv` was generated from `notebooks/00_coco_dataset_exp.ipynb`, so the `file_path` column contains paths like `..\data\coco\images\val2017\000000397133.jpg` — relative to `notebooks/`, not the project root.

**Fix:** Do not use the `file_path` column from metadata to load images. Instead, construct the path from the filename:
```python
IMAGES_DIR = Path("data/coco/images/val2017")
img_path = IMAGES_DIR / row["file_name"]
```

This is what `src/config.py` and `app.py` do.
