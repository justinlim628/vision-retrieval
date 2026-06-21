import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import numpy as np
from PIL import Image

from src.config import INDEX_PATH, METADATA_PATH, IMAGES_DIR, DEFAULT_K, LABEL_K, EMBEDDINGS_PATH, CATEGORIES
from src.clip_model import load_model, encode_text, encode_image
from src.search import load_index, search
from src.labeling import pseudo_label
from src.curation import load_curation_data, get_cluster_choices

print("Loading CLIP model...")
model, preprocess, tokenizer, device = load_model()

print("Loading FAISS index...")
index, metadata = load_index(INDEX_PATH, METADATA_PATH)

print("Computing curation data...")
cluster_labels, outlier_scores = load_curation_data(EMBEDDINGS_PATH, index, metadata)
cluster_choices = get_cluster_choices(cluster_labels, metadata)

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


def run_clustering(cluster_choice, k: int):
    if cluster_choice is None:
        return []

    cid = int(cluster_choice)
    mask = cluster_labels == cid
    rows = metadata[mask]
    sample = rows.sample(n=min(int(k), len(rows)), random_state=0)

    images = []
    for _, row in sample.iterrows():
        img_path = IMAGES_DIR / row['file_name']
        try:
            img = Image.open(img_path).convert('RGB')
            primary = ast.literal_eval(row['labels'])[0]
            images.append((img, f"{primary} · cluster {cid}"))
        except Exception:
            pass

    return images


def run_outliers(top_n: int, category: str):
    if category == 'All':
        indices = np.argsort(outlier_scores)[::-1][:int(top_n)]
        rows = metadata.iloc[indices]
        scores = outlier_scores[indices]
    else:
        mask = metadata['labels'].apply(lambda s: category in ast.literal_eval(s))
        sub_meta = metadata[mask]
        sub_scores = outlier_scores[mask.values]
        order = np.argsort(sub_scores)[::-1][:int(top_n)]
        rows = sub_meta.iloc[order]
        scores = sub_scores[order]

    images = []
    for (_, row), score in zip(rows.iterrows(), scores):
        img_path = IMAGES_DIR / row['file_name']
        try:
            img = Image.open(img_path).convert('RGB')
            primary = ast.literal_eval(row['labels'])[0]
            images.append((img, f"score: {score:.3f} · {primary}"))
        except Exception:
            pass

    return images


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

    with gr.Tab("Clustering"):
        gr.Markdown("Browse COCO images grouped by visual cluster (KMeans on CLIP embeddings).")

        with gr.Row():
            cluster_dropdown = gr.Dropdown(
                choices=cluster_choices,
                label="Cluster",
                value=cluster_choices[0][1],
                scale=4,
            )
            cluster_k_slider = gr.Slider(minimum=1, maximum=30, value=20, step=1, label="Images to show")
            cluster_btn = gr.Button("Browse", variant="primary")

        cluster_gallery = gr.Gallery(label="Cluster images", columns=4, height="auto", object_fit="cover")

        cluster_btn.click(fn=run_clustering, inputs=[cluster_dropdown, cluster_k_slider], outputs=cluster_gallery)

    with gr.Tab("Outliers"):
        gr.Markdown("Find visually atypical images — far from their cluster centroid and sparsely neighbored.")

        with gr.Row():
            outlier_n_slider = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Top N outliers")
            category_dropdown = gr.Dropdown(
                choices=['All'] + CATEGORIES,
                label="Category",
                value='All',
            )
            outlier_btn = gr.Button("Show", variant="primary")

        outlier_gallery = gr.Gallery(label="Outlier images", columns=4, height="auto", object_fit="cover")

        outlier_btn.click(fn=run_outliers, inputs=[outlier_n_slider, category_dropdown], outputs=outlier_gallery)

if __name__ == "__main__":
    demo.launch()
