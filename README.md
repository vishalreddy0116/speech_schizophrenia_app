# Speech-Based Schizophrenia Severity Estimation System Dashboard

A full-stack diagnostic evaluation application implementing the multi-branch VQ-VAE + SDVAE + WavLM Attention-Based Feature Fusion model architecture. This tool allows clinicians or researchers to upload speech recordings or record audio live, execute each step of the processing pipeline, and evaluate severity metrics directly from an interactive web panel.

## System Architecture Coverage
Every component of the proposed system architecture is represented as a clean, granular backend endpoint and visible in the progress tracker:
1. **Data Storage & Management**: Registering raw files & patient metadata in MongoDB (`POST /api/upload`).
2. **Preprocessing & Feature Extraction**: Normalize, pre-emphasis, and fixed 3-second segmenting (`POST /api/preprocess`).
3. **Feature Extraction (4D Articulatory Vector)**: Extract Pitch Variation via YIN, pause ratio, speech rate/ZCR, and articulation (`POST /api/extract`).
4. **Latent Speech Representation Learning**: Translate feature vectors into VQ-VAE (128D) and SDVAE (128D) latent distributions, and WavLM self-supervised sequence embeddings (`POST /api/latent`).
5. **Attention-Based Feature Fusion**: Fuse latent features with WavLM sequence using Multi-Head Attention (`POST /api/fuse`).
6. **MLP Regression Head & Output**: Run fully connected MLP prediction layer to compute final Speech Severity Index (SSI) on a `[0 - 1]` scale and map to a clinical PANSS score (`POST /api/predict`).

---

## Workspace Setup Recommendation

To start working directly with this project, it is highly recommended to configure this folder as your active workspace:
- **Workspace Path**: `C:\Users\visha\.gemini\antigravity-ide\scratch\speech_schizophrenia_app`

---

## Installation & Launch

1. **Install Dependencies**:
   Ensure you have PyTorch, FastAPI, Uvicorn, Librosa, and HuggingFace Transformers installed.
   ```bash
   pip install fastapi uvicorn librosa soundfile transformers torch numpy pandas scikit-learn pymongo
   ```

2. **Start MongoDB (Optional)**:
   The application tries to connect to MongoDB at `mongodb://localhost:27017` on start. If MongoDB is not running, the backend automatically falls back to a local SQLite database (`sqlite_fallback.db`) so the dashboard remains fully functional out-of-the-box.

3. **Run the Application**:
   Execute the launcher script:
   ```bash
   python run.py
   ```

4. **Access the Dashboard**:
   Open your browser and navigate to:
   - **Dashboard UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   - **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
