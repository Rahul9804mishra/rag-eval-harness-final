"""
The 4 retrieval strategies benchmarked in this project:

1. simple_chunking   - fixed-size character splitting (the baseline)
2. semantic_chunking - splits on meaning boundaries
3. hybrid_search     - BM25 keyword search fused with vector similarity
4. reranking         - hybrid search + cross-encoder reranking (best accuracy)

Each module exposes a `build_*_index(...)` function and a
`retrieve_*(...)` function so eval/run_eval.py can call them uniformly.
"""
