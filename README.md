# RAG Evaluation Harness

A self-contained Python benchmark that measures four retrieval strategies
on a 50-question ground-truth dataset and reports Hit Rate, MRR, Recall@K,
and Precision@K. A GitHub Actions workflow fails the build if the best
strategy's accuracy drops below a configurable threshold.

---

## Project Architecture

```
rag-eval-harness/
├── data/
│   └── documents/                  ← source .txt / .pdf files
│       ├── handbook.txt            (original demo — 2 facts)
│       ├── novacorp_employee_handbook.txt   (synthetic demo corpus)
│       ├── novacorp_it_policy.txt           (synthetic demo corpus)
│       └── novacorp_benefits_guide.txt      (synthetic demo corpus)
├── eval/
│   ├── ground_truth.csv            ← 50 Q&A pairs
│   ├── run_eval.py                 ← main evaluation entry point
│   ├── validate_ground_truth.py    ← pre-run sanity check
│   └── check_threshold.py          ← CI pass/fail gate
├── src/
│   ├── config.py                   ← all tunable settings
│   ├── ingest.py                   ← document loader
│   ├── embed_store.py              ← shared HuggingFace embedding model
│   ├── metrics.py                  ← Hit@K, MRR, Recall@K, Precision@K
│   └── strategies/
│       ├── simple_chunking.py      ← Strategy 1 (baseline)
│       ├── semantic_chunking.py    ← Strategy 2
│       ├── hybrid_search.py        ← Strategy 3 (BM25 + vector)
│       └── reranking.py            ← Strategy 4 (cross-encoder)
├── results/                        ← timestamped CSV + JSON per run
├── scripts/
│   └── plot_results.py             ← bar chart generator
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_ground_truth.py
│   ├── test_ingest.py
│   ├── test_strategies.py
│   └── test_results_export.py
├── .github/workflows/eval.yml      ← CI pipeline
├── requirements.txt
└── results.json                    ← latest flat summary (auto-generated)
```

### Pipeline flow

```
Source documents (.txt/.pdf)
        ↓
    ingest.py  →  LangChain Documents
        ↓
 Chunking layer (strategy-specific)
        ↓
  Chroma vector store + BM25 index
        ↓
  Retrieval (top-K chunks per query)
        ↓
  metrics.py  →  Hit@K, MRR, Recall@K, Precision@K
        ↓
  results/eval_<timestamp>.csv
  results/summary_<timestamp>.json
  results.json  (legacy flat summary)
```

---

## Supported Python Version

Python **3.10** or **3.11** recommended. Python 3.12 should also work
but is less tested with this dependency set.

---

## Setup (Windows / PowerShell)

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
# venv\Scripts\activate.bat     # CMD fallback

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the
> `all-MiniLM-L6-v2` embedding model (~90 MB) and the
> `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (~85 MB) on first
> run. Both are cached locally afterwards. No API key is required.

---

## Source Documents

Place `.txt` or `.pdf` files in `data/documents/`. The project ships
with three synthetic demonstration documents:

| File | Contents |
|------|----------|
| `novacorp_employee_handbook.txt` | Employment, leave, performance, conduct policies |
| `novacorp_it_policy.txt` | Device management, access control, incident response |
| `novacorp_benefits_guide.txt` | Health insurance, 401(k), wellness, family benefits |

**These are entirely fictional** and exist only to demonstrate the
pipeline. Replace them (and update `eval/ground_truth.csv`) when you
have real documents to evaluate against.

---

## Running the Evaluation

```powershell
# From the project root with the venv active:

# Optional — validate ground truth first
python eval/validate_ground_truth.py

# Run full evaluation (builds indexes, scores all 50 questions × 4 strategies)
python eval/run_eval.py

# Check CI threshold (exits non-zero if best accuracy < ACCURACY_THRESHOLD)
python eval/check_threshold.py

# Generate bar chart
python scripts/plot_results.py
```

The evaluation prints a table like:

```
============================================================
  Evaluation Results  (50 questions, Top-5)
============================================================
Strategy               Hit%    MRR  Rec%  Pre%
------------------------------------------------------------
simple_chunking        78.0% 0.6800  89.0%  32.0%
semantic_chunking      84.0% 0.7300  91.0%  36.0%
hybrid_search          88.0% 0.7900  93.0%  40.0%
reranking              90.0% 0.8200  94.0%  42.0%
============================================================
```

*(Actual numbers will vary depending on chunking randomness and model
caching state.)*

---

## Running Automated Tests

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a single test file
pytest tests/test_metrics.py -v
```

Tests do **not** require paid APIs. The strategy smoke tests
(`test_strategies.py`) will download the embedding and reranker models
if they are not already cached (~175 MB total, one-time).

---

## Metrics Explained

| Metric | Definition |
|--------|-----------|
| **Hit Rate / Success@K** | Fraction of questions where the exact expected answer appeared in any of the top-K retrieved chunks. Primary benchmark metric. |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank across questions, where rank is the position of the first chunk containing the answer. Rewards finding the answer earlier in the list. |
| **Avg Recall@K** | Token-level recall: what fraction of expected-answer tokens appear anywhere across the top-K chunks? Useful when the answer might be spread. |
| **Avg Precision@K** | Fraction of the top-K chunks that contain the expected answer. Measures how focused the retrieval is. |

All metrics use **exact-match substring checking** (case-insensitive).
They measure retrieval quality, not generated-answer quality.

---

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Latest flat `{strategy: hit_rate_pct}` dict. Used by `check_threshold.py` and `plot_results.py`. |
| `results/eval_<timestamp>.csv` | Full per-question breakdown: strategy, question, expected answer, all four metric scores, first 120 chars of each retrieved chunk. |
| `results/summary_<timestamp>.json` | Per-strategy aggregate metrics for the run. |
| `results_chart.png` | Bar chart generated by `plot_results.py`. |

---

## Configuration (`src/config.py`)

| Setting | Default | Effect |
|---------|---------|--------|
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `SIMPLE_CHUNK_SIZE` | `400` | Characters per chunk (simple strategy) |
| `SIMPLE_CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `HYBRID_ALPHA` | `0.6` | Vector weight in fused score (0=BM25 only, 1=vector only) |
| `RERANK_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for strategy 4 |
| `ACCURACY_THRESHOLD` | `60.0` | Minimum Hit Rate % for CI to pass |

---

## Current Limitations

- **Exact-match scoring only.** A retrieved chunk that paraphrases the
  expected answer will score as a miss. Semantic answer matching would
  require an LLM-as-judge, which adds cost and latency.
- **Local CPU inference.** Reranking on CPU is measurably slower than
  on GPU. The candidate pool is capped at 20 to keep CI times
  reasonable.
- **Synthetic corpus.** The bundled demo documents are fictional. Real
  evaluation requires real documents and carefully written ground truth.
- **No answer generation.** The harness tests retrieval only. Adding a
  generation layer (e.g., with Ollama or an OpenAI model) is left as
  an extension.

---

## Extending the Project

- **Strategy 5: HyDE** — Generate a hypothetical answer first, embed
  it, then retrieve by similarity to that embedding.
- **LLM-as-judge scoring** — Pass retrieved chunks + question to a
  small local LLM and score semantic correctness.
- **Streamlit dashboard** — Display per-question results interactively.
- **Latency tracking** — Add wall-clock timing to show the
  accuracy/speed tradeoff across strategies.
