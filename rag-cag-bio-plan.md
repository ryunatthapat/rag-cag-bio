# RAG & CAG Biographies Demo CLI: Implementation Plan

## Overview
This plan details the step-by-step implementation of a CLI demo that answers user queries about biographies using both Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG). The CLI will display both answers and timings side-by-side for comparison. The design is modular and extensible for future datasets or knowledge sources.

---

## 1. Project Structure

```
rag-cag-bio-demo/
├── cli/                  # CLI entrypoint and user interaction
├── rag/                  # RAG module (vector DB, retrieval, OpenAI)
├── cag/                  # CAG module (KV cache, HuggingFace, torch)
├── data/                 # Data files (biographies.md)
├── utils/                # Shared utilities (data loading, logging, config)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── main.py               # Main CLI entrypoint
```

- **Extensibility:** Structure code as a package, with clear module boundaries and config-driven data/model selection.

---

## 2. Environment & Dependencies
- [ ] Create `requirements.txt` with:
  - torch, transformers
  - qdrant-client (or chromadb)
  - openai
  - tqdm, colorama (for CLI UX)
  - python-dotenv (for env vars)
  - any other required libraries
- [ ] Document environment variable usage (OpenAI API key, HF token, etc.) in `README.md`

---

## 3. Data Preparation
- [ ] Place `biographies.md` in `data/`
- [ ] Implement a data loader in `utils/`:
  - [ ] For RAG: Chunk `biographies.md` by page (using markdown page delimiters)
  - [ ] For CAG: Load the entire file as a single string
- [ ] (Optional) Add config for future data sources

---

## 4. RAG Module (`rag/`)
- [ ] Implement Qdrant (or Chroma) vector DB setup (persisted to disk)
- [ ] Add embedding pipeline using OpenAI (default: text-embedding-ada-002, configurable)
- [ ] On startup:
  - [ ] Chunk `biographies.md` by page
  - [ ] Embed and index each page as a single document in the vector DB
- [ ] Implement retrieval: given a query, return the most relevant page as context
- [ ] Generate answer using OpenAI LLM (gpt-3.5-turbo or gpt-4)
- [ ] Log timing for retrieval and answer generation
- [ ] Expose a function: `rag_answer(query) -> (answer, timing)`

---

## 5. CAG Module (`cag/`)
- [ ] Implement CAG using HuggingFace, torch, and DynamicCache (CPU/GPU as available)
- [ ] On startup:
  - [ ] Load the entire `biographies.md` as a single context
  - [ ] Preprocess into a KV cache (using prompt style as in `cache_prep.py`)
  - [ ] Save/load cache to disk for reuse
- [ ] For each query:
  - [ ] Clean up the cache to original length
  - [ ] Append the question to the prompt and generate answer using the cache
- [ ] Log timing for answer generation
- [ ] Expose a function: `cag_answer(query) -> (answer, timing)`

---

## 6. CLI Interface (`cli/` and `main.py`)
- [ ] Implement a CLI loop for user queries
- [ ] For each query:
  - [ ] Call both `rag_answer(query)` and `cag_answer(query)`
  - [ ] Display both answers and timings side-by-side, with color/highlighting for clarity
  - [ ] Log query, answers, timings, and module used (RAG/CAG) to console
- [ ] Handle errors (API fail, no result, etc.) with user-friendly messages
- [ ] (Optional) Add CLI command to print/export basic stats (average response time, etc.)

---

## 7. Logging & UX
- [ ] Console logging only (no file logging required for demo)
- [ ] Log for each query:
  - Query text
  - RAG answer, timing
  - CAG answer, timing
- [ ] Use color/highlighting for module names and timings

---

## 8. Extensibility
- [ ] Structure code for easy extension:
  - Configurable data/model paths
  - Modular pipeline (easy to swap in new datasets or models)
  - Document how to add new knowledge sources

---

## 9. Testing & Validation
- [ ] Manual testing for all flows (RAG, CAG, CLI, error handling)
- [ ] Validate CPU-only operation on Mac (Apple Silicon/Intel)
- [ ] (Optional) Add basic unit/integration tests for core modules

---

## 10. Documentation
- [ ] Write a clear `README.md` with setup, usage, and architecture overview
- [ ] Document all environment variables and configuration options
- [ ] Add code comments and docstrings for maintainability

---

## 11. (Optional) Dockerization & Deployment
- [ ] Add a `Dockerfile` for local and containerized development (CPU by default)
- [ ] Document how to build and run the system locally and in Docker

---

## 12. Example CLI Flow

```
> Who is John Doe?

[RAG]   (1.23s): John Doe is a software engineer at ...
[CAG]   (2.01s): John Doe is a ...
```

---

## 13. Immediate Next Steps
- [ ] Set up project structure and requirements
- [ ] Implement data loaders and chunking
- [ ] Build RAG and CAG modules as described
- [ ] Implement CLI and logging
- [ ] Test end-to-end flow
