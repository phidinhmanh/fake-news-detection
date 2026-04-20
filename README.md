# Verity — Advanced Vietnamese Fake News Detection
🛡️ 🔍 🚀

**Verity** is a concise, high-performance fake news detection system specifically optimized for the Vietnamese language. It integrates **PhoBERT** deep learning baselines with a state-of-the-art **8-stage Sequential Adversarial LLM Pipeline** for explainable fact-checking.

---

## 🏗️ Project Architecture (Post-Refactor)

The project has been refactored to remove 70% of redundant code, leaving a focused 30% core:

1.  **`sequential_adversarial/`**: The core 8-stage multi-agent pipeline (Lead Investigator, Data Analyst, Bias Auditor, etc.).
2.  **`dataset/`**: Vietnamese-centric data pipeline (ViFactCheck, ReINTEL) using `underthesea`.
3.  **`model/`**: Training scripts for PhoBERT and the TF-IDF baseline.
4.  **`agents/`**: RAG infrastructure (LanceDB + Evidence Searcher).
5.  **`ui/`**: Modern Streamlit interface.
6.  **`evaluation/`**: Consolidated scripts for model comparison.
7.  **`saved_models/`**: Storage for trained artifacts.

---

## 🚀 How to Run

### 1. Setup Environment
Requires `uv` package manager.
```bash
git clone ...
cd fake-news-detection
uv sync
cp .env.example .env  # Add your GOOGLE_API_KEY, NVIDIA_API_KEY, etc.
```

### 2. Data Preparation
The consolidated dataset manager handles the entire pipeline (Download -> Preprocess -> features) in one command:
```bash
uv run python dataset/manager.py --full
```

### 3. Training Baselines
Train the classic ML or Deep Learning baselines:
```bash
uv run python model/baseline_logreg.py  # TF-IDF + LogReg
uv run python model/train_phobert.py    # PhoBERT (with --features support)
```

### 4. Launch the Verity Engine (UI)
```bash
uv run streamlit run ui/app.py
```
Access the dashboard at **http://localhost:8501**.

---

## 🧬 Sequential Adversarial Pipeline (8 Stages)

1.  **Input Processor**: URL/Text extraction.
2.  **Lead Investigator**: Claim & Loaded Language extraction.
3.  **Data Analyst**: RAG-based Fact Checking (Wikipedia/Local DB).
4.  **Bias Auditor**: Adversarial critique of analysis.
5.  **Synthesizer**: Final consensus & markdown report.
6.  **Visual Engine**: Logic flow generation (Mermaid).
7.  **Persistence**: SQL logging.
8.  **TF-IDF Comparator**: Safety check against classic baseline.

---

## 👥 Contributors
UET Group Project — 2026. Optimized for precision, speed, and explainability.
