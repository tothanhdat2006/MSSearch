# Pilot Study: Does CVSearch Realize No Evidence Exists in Images? (N=30)

## 1. Problem Formulation & Research Question
Standard VLM visual search benchmarks (e.g., HR-Bench) always supply an image containing the target evidence. When visual search methods are deployed in unconstrained settings, questions will frequently refer to objects or evidence completely absent from the current image.

**Diagnostic Question**: Does CVSearch ([ICML 2026](https://github.com/liliupeng28/ICML26-CVSearch)) recognize when **no evidence exists in the image**, or does it hallucinate via parametric priors and over-segment irrelevant background textures?

---

## 2. Experimental Design ($N=30$)
We evaluate CVSearch under a strict **Negative Control / Cross-Modal Derangement**:
- For each sample canvas $I_i$ ($i \in \{0, \dots, 29\}$), we randomly sample an alien question $Q_{\pi(i)}$ with options and ground truth from sample $\pi(i) \neq i$.
- **Strict Non-Collision Invariant**:
  $$\forall i \in \{0, \dots, 29\}: \quad i \neq \pi(i), \quad I_i \neq I_{\pi(i)}, \quad Q_i \neq Q_{\pi(i)}$$
- Enforced with deterministic seeding (`seed=42`) and algorithmic assertions.

---

## 3. Evidence Realization Diagnostic Metrics

| Metric | Condition | Meaning |
| :--- | :--- | :--- |
| **Root Prior Hallucination** | $C_{\text{root}} > 0.6$ | Model bypasses visual search entirely based on linguistic priors, falsely claiming high confidence on an alien question. |
| **SAM3 False Detection** | $\sum \text{flags} > 0$ | SAM3 hallucinates bounding boxes for non-existent target objects on the canvas. |
| **Fine Search Hallucination** | $\text{is\_success} = \text{True}$ | Fine search tree erroneously reports finding target evidence in a sub-patch. |
| **Realized Absence** | $C_{\text{root}} \le 0.6 \land \sum \text{flags} = 0 \land \neg \text{is\_success}$ | System correctly discovers that no evidence exists for the target entity. |
| **Donor Accuracy** | $\hat{y} = y_{\text{donor}}$ | Accuracy on donor ground truth; expected to hover at chance level ($\approx 25.0\%$). |

---

## 4. Repository & Path Layout
- **Script Directory**: `MSSearch/pilot/cvsearch_no_evidence_pilot30/`
- **CVSearch Repository**: Automatically resolved with fallback ladder:
  1. CLI argument: `--cvsearch_repo /path/to/ICML26-CVSearch`
  2. Sibling directory at same level: `MSSearch/pilot/ICML26-CVSearch`
  3. Reference workspace path: `docs/experiments/freeform/ICML26-CVSearch`
  4. Kaggle runtime path: `/kaggle/working/ICML26-CVSearch`

---

## 5. Execution

### 1) Run Verification Test (Zero GPU, Standalone Assertions)
```bash
python test_derangement.py
```

### 2) Run Mock Simulation (Pipeline & Metric Aggregation Check)
```bash
python pilot_study.py --mock --n_samples 30
```

### 3) Run Full GPU Inference on Server / Kaggle
```bash
bash run_pilot.sh 30
```
or with explicit paths:
```bash
python pilot_study.py \
  --cvsearch_repo ../ICML26-CVSearch \
  --dataset_path ../ICML26-CVSearch/cvsearch/CVsearch/annotation_hr-bench_4k.json \
  --image_dir ../ICML26-CVSearch/cvsearch/CVsearch/image \
  --qwen_path /path/to/Qwen2.5-VL-7B-Instruct \
  --sam_path /path/to/sam3.pt \
  --n_samples 30
```
