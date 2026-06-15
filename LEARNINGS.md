# Learnings

## 2026-06-13

### Session: Modularize notebook pipeline → Gradio search app

1. **`metadata.csv` file_path column is notebook-relative, not project-relative.**
   The `file_path` column (e.g. `..\data\coco\images\val2017\...`) was written by `notebooks/00_coco_dataset_exp.ipynb` with paths relative to `notebooks/`. Never use this column to load images from outside the notebook. Always construct paths as `IMAGES_DIR / row["file_name"]` where `IMAGES_DIR = Path("data/coco/images/val2017")`.

2. **FAISS index positions map directly to metadata DataFrame rows.**
   `index.search()` returns position indices (0–3140) that correspond directly to `metadata.iloc[i]`. The `embedding_indices.npy` file is not needed for search — it was only saved during the embedding generation step as a record of which DataFrame rows were successfully encoded.

3. **Gradio `StarletteDeprecationWarning` about `HTTP_422_UNPROCESSABLE_ENTITY` is harmless noise.**
   This warning appears in the terminal on every search request due to a Starlette version mismatch with the installed Gradio. It does not affect functionality and can be ignored.

4. **`src/` as a flat folder of modules requires `sys.path` injection in the entrypoint.**
   Since `src/` is not a Python package (no `__init__.py`), `app.py` adds it to `sys.path` via `sys.path.insert(0, str(Path(__file__).parent))` so that `from src.config import ...` resolves correctly when running `python app.py` from the project root.

5. **CLIP model startup takes ~30 seconds on CPU.**
   Loading `ViT-B-32` with `open_clip.create_model_and_transforms()` on first run downloads weights and warms up the model. Subsequent runs are faster (weights are cached locally by HuggingFace Hub).

---

## TODO (next session)

- [ ] Add auto-labeling feature to Gradio app (single-label or multi-label tab)
- [ ] Add clustering / outlier visualization tab to Gradio app
- [ ] Consider adding a `scripts/build_index.py` to regenerate embeddings and FAISS index from scratch
- [ ] Explore adding image upload search (encode image → FAISS search) alongside text search
