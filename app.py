import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from PIL import Image

from src.config import INDEX_PATH, METADATA_PATH, IMAGES_DIR, DEFAULT_K
from src.clip_model import load_model, encode_text
from src.search import load_index, search

print("Loading CLIP model...")
model, tokenizer, device = load_model()

print("Loading FAISS index...")
index, metadata = load_index(INDEX_PATH, METADATA_PATH)

print("Ready.")


def run_search(query: str, k: int):
    if not query.strip():
        return []

    query_embedding = encode_text(query, model, tokenizer, device)
    results = search(query_embedding, index, metadata, k=int(k))

    images = []
    for _, row in results.iterrows():
        img_path = IMAGES_DIR / row["file_name"]
        try:
            img = Image.open(img_path).convert("RGB")
            caption = f"score: {row['score']:.3f}"
            images.append((img, caption))
        except Exception:
            pass

    return images


with gr.Blocks(title="CLIP Image Search") as demo:
    gr.Markdown("## CLIP Semantic Image Search")
    gr.Markdown("Search the COCO dataset using natural language.")

    with gr.Row():
        query_box = gr.Textbox(
            placeholder="e.g. a dog playing outside",
            label="Search query",
            scale=4,
        )
        k_slider = gr.Slider(minimum=1, maximum=20, value=DEFAULT_K, step=1, label="Results (k)")
        search_btn = gr.Button("Search", variant="primary")

    gallery = gr.Gallery(label="Results", columns=3, height="auto", object_fit="cover")

    search_btn.click(fn=run_search, inputs=[query_box, k_slider], outputs=gallery)
    query_box.submit(fn=run_search, inputs=[query_box, k_slider], outputs=gallery)

if __name__ == "__main__":
    demo.launch()
