# VectorDB (Python) - a Vector Database from Scratch

A Python port of the C++ "Your-OWN-AI" project. It implements **HNSW**, **KD-Tree** and
**Brute Force** search side by side, three distance metrics, a 16D demo dataset with a
PCA scatter-plot UI, a document store using real Ollama embeddings, and a **RAG pipeline**
powered by a local LLM.

It uses **only the Python standard library** - nothing to `pip install`.
The web UI (`index.html`) is the original, unchanged.

## Requirements

1. **Python 3.8+**
2. **Ollama** (only for the Documents and Ask AI tabs) - https://ollama.com

```
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Run

```
python main.py
```

Open http://localhost:8080. Ollama is optional: the Search tab works without it, and
the header shows `OFFLINE` until you start `ollama serve`.

## How it works

```
Your text -> Ollama (nomic-embed-text, 768D) -> HNSW index -> semantic search
          -> Ollama (llama3.2) reads retrieved chunks -> answer
```

| Component | Class in `main.py` | Notes |
|---|---|---|
| Brute force | `BruteForce` | exact, O(N·d) baseline |
| KD-Tree | `KDTree` | exact, axis-aligned partitioning; rebuilt on delete |
| HNSW | `HNSW` | multilayer small-world graph, M=16, efConstruction=200, ef search=50 |
| Demo index | `VectorDB` | all three indexes over 20 16D vectors |
| Document index | `DocumentDB` | HNSW (brute force when < 10 chunks) over Ollama embeddings |
| Ollama client | `OllamaClient` | `/api/embeddings` and `/api/generate` via `urllib` |
| Chunker | `chunk_text` | 250-word chunks, 30-word overlap |

## REST API (same as the C++ version)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,...,f16&k=5&metric=cosine&algo=hnsw` | k-NN search (`algo`: hnsw / kdtree / bruteforce; `metric`: cosine / euclidean / manhattan) |
| POST | `/insert` | `{"metadata","category","embedding"}` insert a demo vector |
| DELETE | `/delete/:id` | delete a demo vector |
| GET | `/items` | list demo vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | time all three algorithms |
| GET | `/hnsw-info` | graph structure and per-layer stats |
| GET | `/stats` | database statistics |
| POST | `/doc/insert` | `{"title","text"}` chunk, embed and store |
| GET | `/doc/list` | list stored chunks |
| DELETE | `/doc/delete/:id` | delete a chunk |
| POST | `/doc/search` | `{"question","k"}` retrieval only |
| POST | `/doc/ask` | `{"question","k"}` full RAG: retrieve + generate |
| GET | `/status` | Ollama status and model info |

```
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3"
curl -X POST http://localhost:8080/doc/ask -d '{"question":"What is dynamic programming?","k":3}'
```

## Configuration

- Faster LLM: `ollama pull llama3.2:1b`, then set `self.gen_model = "llama3.2:1b"` in `OllamaClient`.
- Port: change `PORT` in `main.py`. The UI calls `http://localhost:8080`, so also change `API` near the top of the script in `index.html`.

## Differences from the C++ original

- Plain Python floats (64-bit) instead of 32-bit floats; distances can differ in the last digits.
- Python's `random` replaces `std::mt19937`, so the HNSW graph layout differs between the two (same seed, different generator). Search results are equivalent.
- JSON is parsed with the `json` module rather than hand-rolled string scanning.
- `index.html` is served from the script's folder, so you can start `main.py` from any directory.

## Project structure

```
Your-OWN-AI-python/
├── main.py            <- backend: HNSW, KD-Tree, BruteForce, REST API, RAG
├── index.html         <- frontend (unchanged)
├── requirements.txt   <- empty: standard library only
└── README.md
```

## License

MIT
