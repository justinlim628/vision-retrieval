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

## 2026-06-16

### Session: Add auto-label tab (image upload → pseudo-labeling via FAISS neighbors)

1. **`open_clip.create_model_and_transforms()` returns `(model, preprocess, _)` — `preprocess` was being discarded.**
   The middle return value is a torchvision transform that resizes and normalizes images to CLIP's expected input (224×224, ImageNet stats). It is required for `model.encode_image()`. Previously discarded as `_`, it now must be captured and returned from `load_model()` and passed to `encode_image()`.

2. **Weighted vote is a clean multi-label confidence formula: `confidence(c) = Σ score_i [c ∈ labels_i] / Σ score_i`.**
   Confidences are computed independently per category and intentionally don't sum to 100% — an image tagged "car, truck" boosts both. Normalizing by total neighbor score (not count) gives higher weight to more similar images.

3. **`labels` column in metadata.csv is a Python list string, not JSON — use `ast.literal_eval()` to parse it.**
   The column stores values like `"['car', 'truck']"` with single quotes. `json.loads()` would fail; `ast.literal_eval()` handles it correctly.

4. **Feature branch workflow for new features: branch → commit → fast-forward merge → delete branch.**
   `git checkout -b feature/X` → implement → `git commit` → `git checkout main` → `git merge feature/X` (fast-forward when main hasn't diverged) → `git branch -d feature/X`. Keep commits focused on one logical change.

5. **Wrapping existing Gradio UI in tabs requires nesting components inside `with gr.Tab("name"):` blocks.**
   No components need to be recreated — just indent them under `gr.Tab`. Event handlers (`.click()`, `.submit()`) remain inside the same `gr.Tab` block as their components.

---

## TODO (next session)

- [ ] Add clustering / outlier visualization tab to Gradio app
- [ ] Consider adding a `scripts/build_index.py` to regenerate embeddings and FAISS index from scratch
- [ ] Explore image-to-image search tab (encode uploaded image → FAISS search → show visually similar results)
- [ ] Add confidence threshold display note or visual bar chart to auto-label tab (e.g. `gr.BarPlot`) for better readability
