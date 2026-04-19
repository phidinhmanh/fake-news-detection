# Code Review: Duplication & SOLID — Full Codebase
## Review ID: review-20260418-fullcodebase
## Date: 2026-04-18
## Agent: review-cycle (session mode → module mode fallback)

---

## Aggregation Summary

| Dimension | Files | Findings | Critical | High | Medium | Low |
|-----------|-------|----------|----------|------|--------|-----|
| **Architecture** | 14 | 12 | 2 | 6 | 3 | 1 |
| **Quality** | — | 12 | — | — | — | — |
| **Maintainability** | 25 | 47 | 0 | 8 | 22 | 17 |
| **TOTAL** | — | **71** | **2** | **14** | **25** | **18** |

---

## 🔴 CRITICAL Issues (Require Immediate Fix)

### CRIT-1: `Predictor` class (571 lines) — 5+ Responsibilities
- **File:** `model/predictor.py:49`
- **Violation:** SRP (Single Responsibility Principle)
- **Problem:** The class handles: (1) ensemble prediction, (2) domain classification, (3) SHAP/attention explanations, (4) mock fallback, (5) keyword explanations
- **Action:** Split into 4 classes:
  - `PredictionService` — ensemble coordination only
  - `DomainClassifier` — domain classification with keyword fallback
  - `ExplanationGenerator` — SHAP + attention token explanations
  - `MockPredictor` — mock prediction logic

### CRIT-2: DIP Violation — High-Level Depends on Concrete
- **Files:** `model/predictor.py:34`, `ui/app.py:184`
- **Violation:** Dependency Inversion Principle
- **Problem:** `Predictor` depends on concrete `EnsembleClassifier`; UI imports concrete model classes
- **Action:** Define `PredictionInterface` abstract base class; use dependency injection

---

## 🟠 HIGH Priority Issues (Fix in Current Sprint)

### ARCH-1: `EnsembleClassifier` — 3 Responsibilities
- **File:** `model/ensemble.py:46`
- **Violation:** SRP
- **Problem:** Handles prediction, caching, AND optimization
- **Action:** Split into `EnsembleClassifier`, `PredictionCache`, `WeightOptimizer`

### ARCH-2: `train.py` — Mixed Responsibilities
- **File:** `model/train.py:55`
- **Violation:** SRP
- **Problem:** `FakeNewsClassifier` + `CurriculumLearningScheduler` + orchestration in same file
- **Action:** Split into `model/classifier.py`, `model/curriculum_scheduler.py`, `train.py` (orchestration only)

### ARCH-3: Hardcoded Domain Keywords
- **File:** `model/predictor.py:281`
- **Violation:** OCP (Open/Closed Principle)
- **Problem:** Adding new domains requires modifying `Predictor` class
- **Action:** Move to `config.py` or external domain registry

### ARCH-4: Hardcoded Dataset Paths
- **File:** `data/preprocessing.py:78`
- **Violation:** OCP
- **Problem:** Adding new datasets requires modifying `preprocessing.py`
- **Action:** Use factory pattern with dataset registry

### ARCH-5: `ui/app.py` Depends on Concrete Models
- **File:** `ui/app.py:184`
- **Violation:** DIP
- **Problem:** UI directly imports `Predictor` and model classes
- **Action:** Create `PredictionService` interface; inject via Streamlit session state

### MAINT-1: `train_with_curriculum` — 138 Lines
- **File:** `model/train.py:375`
- **Category:** Long method
- **Action:** Extract curriculum stage logic into `CurriculumScheduler`

### MAINT-2: `_attention_explain` — 107 Lines
- **File:** `model/predictor.py:364`
- **Category:** Long method
- **Action:** Extract token aggregation into `AttentionExplainer` helper class

### MAINT-3: Magic Number `max_features=5000` Duplicated
- **Files:** `model/baseline_logreg.py:19`, `train_cv_simple.py:70`, `train_cv_comparison.py:79`
- **Category:** Duplication + magic number
- **Action:** Define `TFIDF_MAX_FEATURES = 5000` in `config.py`

---

## 🟡 MEDIUM Priority (Refactor When Touching)

| ID | Issue | File | Category |
|----|-------|------|----------|
| ARCH-6 | `ui/app.py` mixes UI + input + analysis | `ui/app.py:25` | SRP |
| ARCH-7 | `pipeline.py` has 8 stage classes + orchestrator | `sequential_adversarial/pipeline.py:44` | SRP |
| ARCH-8 | `llm_client.py` has 7 LLM providers | `sequential_adversarial/llm_client.py:136` | SRP |
| MAINT-4 | Hardcoded `API_PORT = 8000` | `config.py:47` | Magic number |
| MAINT-5 | `MAX_TEXT_LENGTH = 2048` hardcoded | `config.py:52` | Magic number |
| MAINT-6 | `DEFAULT_MAX_SEQ_LEN = 256` misalignment | `config.py:61` | Magic number |

---

## 🟢 LOW Priority (Cosmetic / Future)

| ID | Issue | File |
|----|-------|------|
| ARCH-9 | `EnsembleClassifier` exposes 15+ methods | `model/ensemble.py:179` |
| ARCH-10 | LLM providers have inconsistent mock behavior | `sequential_adversarial/llm_client.py:203` |

---

## Refactoring Roadmap

### Phase 1: Critical Fixes (1-2 days)
1. Extract `PredictionService` interface in `api/schemas.py` or new `model/interfaces.py`
2. Split `Predictor` class into 4 focused classes
3. Update `ui/app.py` to use interface via dependency injection

### Phase 2: High Priority (3-4 days)
4. Split `EnsembleClassifier` (prediction / cache / optimizer)
5. Move domain keywords → `config.py` domain registry
6. Move dataset paths → factory pattern in `data/`
7. Extract `CurriculumLearningScheduler` from `train.py`

### Phase 3: Medium Priority (1 week)
8. Split `pipeline.py` into per-stage modules
9. Split `llm_client.py` into per-provider modules
10. Extract `_attention_explain` → `AttentionExplainer` helper
11. Consolidate magic numbers in `config.py`

### Phase 4: Low Priority (Backlog)
12. Refine `EnsembleClassifier` interface (reduce to <10 methods)
13. Standardize LLM mock behavior across providers

---

## Files Requiring Changes

```
model/predictor.py          # CRIT-1, ARCH-3, MAINT-2
model/ensemble.py           # ARCH-1, ARCH-9
model/train.py              # ARCH-2, MAINT-1
config.py                   # ARCH-3, MAINT-4, MAINT-5, MAINT-6
data/preprocessing.py       # ARCH-4
ui/app.py                   # ARCH-2, ARCH-5, ARCH-6
sequential_adversarial/pipeline.py    # ARCH-7
sequential_adversarial/llm_client.py  # ARCH-8, ARCH-10
```

**Total: 8 files need modification**