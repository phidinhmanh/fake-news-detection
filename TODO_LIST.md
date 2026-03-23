# Fake News Detection — TODO List (Team Tracking)

Hệ thống quản lý task cho 3 người. Cập nhật trạng thái tại đây sau mỗi PR.

---

## 🚀 IMPL-000: Day-1 Setup (Cả 3 người)
- [x] IMPL-000.1: Scaffold thư mục + gitignore (Người A)
- [x] IMPL-000.2: schemas.py — API Contract (Người A)
- [x] IMPL-000.3: mock_server.py (Người A)
- [x] IMPL-000.4: requirements.txt (Người A)
- [x] IMPL-000.5: config.py (Người A)
- [x] IMPL-000.6: Makefile (Người A)
- [x] IMPL-000.7: tests/test_schemas.py (Người A)
- [x] IMPL-000.8: tests/test_mock_server.py (Người A)
- [x] IMPL-000.9: README.md skeleton (Người A)

---

## 📂 IMPL-001: Data Pipeline — Tuần 1-2 (Người A)
- [ ] IMPL-001.1: Crawler VnExpress
- [ ] IMPL-001.2: Crawler Tuổi Trẻ
- [ ] IMPL-001.3: Crawler Reuters
- [ ] IMPL-001.4: Load VFND + FakeNewsNet
- [ ] IMPL-001.5: DataModule v1 (PyTorch Lightning)
- [ ] IMPL-001.6: Preprocessing pipeline

## 🤖 IMPL-002: Baseline Model — Tuần 1-2 (Người B)
- [ ] IMPL-002.1: HuggingFace + Lightning training loop
- [ ] IMPL-002.2: TF-IDF + LogReg baseline
- [ ] IMPL-002.3: Metrics logging (F1 ≥ 0.72)
- [ ] IMPL-002.4: predictor.py v1 (mock interface)

## 🎨 IMPL-003: UI Skeleton — Tuần 1-2 (Người C)
- [ ] IMPL-003.1: Layout skeleton app.py
- [ ] IMPL-003.2: Connect mock server
- [ ] IMPL-003.3: Basic result rendering
- [ ] IMPL-003.4: Custom CSS base styles

---

## 📈 IMPL-004: Data Augmentation — Tuần 3-4 (Người A)
- [ ] IMPL-004.1: Back-translation tiếng Việt
- [ ] IMPL-004.2: Class balancing WeightedRandomSampler
- [ ] IMPL-004.3: Data quality validation

## 🧠 IMPL-005: LoRA Fine-tune — Tuần 3-4 (Người B)
- [ ] IMPL-005.1: LoRA fine-tune XLM-RoBERTa-base
- [ ] IMPL-005.2: Curriculum learning (easy → hard)
- [ ] IMPL-005.3: Evaluation pipeline (F1 ≥ 0.85)

## 📊 IMPL-006: Prediction UI — Tuần 3-4 (Người C)
- [ ] IMPL-006.1: Prediction score bar + confidence gauge
- [ ] IMPL-006.2: SHAP token highlight
- [ ] IMPL-006.3: Result layout refinement

---

## 🔗 IMPL-007: Domain Classifier + Ensemble — Tuần 5-6 (Người A + B)
- [ ] IMPL-007.1: Domain classifier (Người A)
- [ ] IMPL-007.2: Ensemble: LLM + rule-based (Người B)
- [ ] IMPL-007.3: SHAP integration (Người B)
- [ ] IMPL-007.4: predictor.py final interface (Người B)

## 💎 IMPL-008: Full UI — Tuần 5-6 (Người C)
- [ ] IMPL-008.1: Domain badge component
- [ ] IMPL-008.2: Source credibility panel
- [ ] IMPL-008.3: History log + CSV export
- [ ] IMPL-008.4: Full app.py v2

---

## 🌍 IMPL-009: Integration & Deploy — Tuần 7 (Cả 3 người)
- [ ] IMPL-009.1: Swap mock → model thật
- [ ] IMPL-009.2: Integration test toàn bộ (test_pipeline.py)
- [ ] IMPL-009.3: Deploy Streamlit Cloud
- [ ] IMPL-009.4: README.md hoàn chỉnh
- [ ] IMPL-009.5: Demo prep (slides + live demo)
