import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from PIL import Image

from src.config import INDEX_PATH, METADATA_PATH, IMAGES_DIR, DEFAULT_K, LABEL_K
from src.clip_model import load_model, encode_text, encode_image
from src.search import load_index, search
from src.labeling import pseudo_label

print("Loading CLIP model...")
model, preprocess, tokenizer, device = load_model()

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


def run_label(image):
    if image is None:
        return [], []

    query_embedding = encode_image(image, model, preprocess, device)
    results = search(query_embedding, index, metadata, k=LABEL_K)

    label_results = pseudo_label(results)
    table = [[r['label'], f"{r['confidence']:.1%}"] for r in label_results]

    images = []
    for _, row in results.iterrows():
        img_path = IMAGES_DIR / row['file_name']
        try:
            img = Image.open(img_path).convert('RGB')
            images.append((img, f"score: {row['score']:.3f}"))
        except Exception:
            pass

    return table, images


with gr.Blocks(title="CLIP Image Search") as demo:
    gr.Markdown("## CLIP Semantic Image Search")

    with gr.Tab("Search"):
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

    with gr.Tab("Auto-Label"):
        gr.Markdown("Upload an image to predict its labels from similar images in the database.")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload image")
            label_btn = gr.Button("Label", variant="primary")

        label_table = gr.Dataframe(headers=["Label", "Confidence"], label="Predicted labels")
        similar_gallery = gr.Gallery(label="Similar images used for labeling", columns=5, height="auto", object_fit="cover")

        label_btn.click(fn=run_label, inputs=[image_input], outputs=[label_table, similar_gallery])

if __name__ == "__main__":
    demo.launch()
