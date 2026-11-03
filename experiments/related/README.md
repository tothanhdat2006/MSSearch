# Related Multi-Source Visual Search Approaches

Architectural blueprints and integration contracts for related iterative and tree-search visual search baselines adapted to the multi-source candidate setting ($N \in \{5, 10, 20\}$).

---

## 1. Targeted Related Methods

### 1.1 CVSearch (Compositional Visual Search)
- **Original Focus**: Single high-resolution image retrieval and query composition.
- **Multi-Source Adaptation**:
  - Independent image relevance scoring via compositional query embeddings.
  - Top-$K$ candidate image filtering followed by coarse-to-fine visual question answering.
- **Evaluation Contract**: Implements `eval_runner.py` interface with `--eval_strategy cvsearch`.

### 1.2 TreeSearch (Hierarchical Coarse-to-Fine Search)
- **Original Focus**: Recursive quadtree spatial decomposition on single ultra-high-resolution images.
- **Multi-Source Adaptation**:
  - Root forest node with $N$ candidate trees.
  - Priority-queue driven best-first search across all $N$ trees with unified visual token budget.
- **Evaluation Contract**: Implements `eval_runner.py` interface with `--eval_strategy treesearch`.

### 1.3 V* (Guided Visual Search via LLM + MLLM)
- **Original Focus**: Active visual search utilizing an LLM guiding a vision-language model across bounding box crops.
- **Multi-Source Adaptation**:
  - Multi-image prompt memory maintaining candidate image identities.
  - Action space extended to `[SWITCH_IMAGE(id), ZOOM_IN(bbox), READ_ANSWER]`.
- **Evaluation Contract**: Implements `eval_runner.py` interface with `--eval_strategy vstar`.

---

## 2. Common Evaluation Contract

All related methods must conform to the unified evaluation input/output schema:

- **Input**:
  - `candidate_images: List[PIL.Image.Image]` ($N \in [5, 20]$)
  - `question: str`
  - `budget_max_patches: int` (proportional to candidate set size $N$)
- **Output Schema**:
  ```python
  {
      "primary_answer": str,
      "prediction_list": List[str],
      "discovered_evidence_images": List[str],
      "telemetry": {
          "route": str,
          "timings_sec": Dict[str, float]
      }
  }
  ```

---

## 3. Directory Layout (Scaffolding)

```
experiments/related/
├── README.md               # Architectural specifications and adaptation notes
├── cvsearch/               # CVSearch adapter and run scripts (planned)
├── treesearch/             # TreeSearch adapter and run scripts (planned)
└── vstar/                  # V* adapter and run scripts (planned)
```
