# VectorMind AI

**An intelligent vector database and Retrieval-Augmented Generation
(RAG) engine built from scratch in Python.**

[![Live
Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render&logoColor=white)](https://vectormind-ai-nsg8.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** https://vectormind-ai-nsg8.onrender.com/

VectorMind AI is a Python implementation of a vector database. It lets
you compare **HNSW, KD-Tree, and Brute Force** nearest-neighbour search
using multiple distance metrics. A browser-based dashboard provides
vector search, PCA-based visualization, and algorithm benchmarking.

It also includes a document store and a Retrieval-Augmented Generation
(RAG) workflow using Ollama for embeddings and text generation when an
Ollama service is available.

## Features

-   **HNSW search:** approximate nearest-neighbour search using a
    multilayer graph.
-   **KD-Tree search:** exact search using axis-aligned partitioning.
-   **Brute Force search:** exact nearest-neighbour baseline.
-   **Distance metrics:** Cosine, Euclidean, and Manhattan.
-   **Interactive dashboard:** search vectors and inspect results in the
    browser.
-   **PCA visualization:** explore the demo vector dataset in a 2D
    scatter plot.
-   **Benchmarking:** compare the three search algorithms for the same
    query.
-   **Document store:** split documents into chunks and store their
    embeddings.
-   **RAG pipeline:** retrieve relevant chunks and use an Ollama
    language model to generate an answer.
-   **REST API:** endpoints for search, indexing, benchmarking,
    documents, and status.

## Technology Stack

-   Python (standard library)
-   HTML, CSS, and JavaScript
-   HNSW, KD-Tree, and Brute Force
-   Ollama (`nomic-embed-text` and `llama3.2`)
-   Render (deployment)

The core Python application uses the standard library, so no third-party
Python packages are required by the original implementation. The
`requirements.txt` file may be empty.

## Live Demo

**[Open VectorMind AI](https://vectormind-ai-nsg8.onrender.com/)**

The hosted application uses Render's free service tier. Free services
may spin down after inactivity, so the first visit after a period
without traffic may take longer.

### Ollama availability

The document-embedding and RAG features depend on an accessible Ollama
service. In the local setup, the application connects to Ollama running
on your computer. A cloud deployment cannot automatically access Ollama
on your personal computer.

Therefore, the **Documents** and **Ask AI** features require Ollama to
be configured and reachable from the deployed backend. Do not assume
those features work in the public demo unless a remote Ollama-compatible
service has been configured. The vector-search demo does not require
Ollama.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/lohitakshajoshi28/VectorMind-AI.git
cd VectorMind-AI
```

If the application files are inside a subfolder, change into the
directory containing `main.py` and `index.html`.

### 2. Check Python

Python 3.8 or later is required.

``` bash
python --version
```

### 3. Start the application

``` bash
python main.py
```

Open **http://localhost:8080** in your browser.

The core vector-search functionality can run without Ollama.

### 4. Enable document embeddings and RAG (optional)

Install [Ollama](https://ollama.com), then download the models:

``` bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Make sure Ollama is running before using document embedding and Ask AI
features.

## How It Works

The document workflow:

``` text
Document text
    |
    v
Text chunking
    |
    v
Ollama embedding model (nomic-embed-text)
    |
    v
Vector embeddings
    |
    v
HNSW document index
    |
    v
Retrieve relevant chunks
    |
    v
Ollama language model (llama3.2)
    |
    v
Generated answer
```

The demo vector dataset can be searched using HNSW, KD-Tree, or Brute
Force. The benchmark endpoint measures the algorithms for a supplied
query.

## Search Components

  -----------------------------------------------------------------------
  Component               Class                   Description
  ----------------------- ----------------------- -----------------------
  Brute Force             `BruteForce`            Exact nearest-neighbour
                                                  baseline that scans the
                                                  vectors.

  KD-Tree                 `KDTree`                Exact search using
                                                  axis-aligned
                                                  partitioning; rebuilt
                                                  on deletion.

  HNSW                    `HNSW`                  Multilayer small-world
                                                  graph for approximate
                                                  nearest-neighbour
                                                  search.

  Demo index              `VectorDB`              Runs the three indexes
                                                  over the demo vector
                                                  dataset.

  Document index          `DocumentDB`            Stores document chunks
                                                  and their Ollama
                                                  embeddings.

  Ollama client           `OllamaClient`          Calls Ollama embedding
                                                  and generation APIs.

  Text chunker            `chunk_text`            Splits text into
                                                  250-word chunks with
                                                  30-word overlap.
  -----------------------------------------------------------------------

The original HNSW configuration uses `M=16`, `efConstruction=200`, and
an `ef` search value of `50`.

## REST API

  ----------------------------------------------------------------------------------------------------
  Method                  Endpoint                                             Description
  ----------------------- ---------------------------------------------------- -----------------------
  `GET`                   `/search?v=f1,...,f16&k=5&metric=cosine&algo=hnsw`   k-nearest-neighbour
                                                                               search. `algo`: `hnsw`,
                                                                               `kdtree`, or
                                                                               `bruteforce`; `metric`:
                                                                               `cosine`, `euclidean`,
                                                                               or `manhattan`.

  `POST`                  `/insert`                                            Insert a demo vector
                                                                               using `metadata`,
                                                                               `category`, and
                                                                               `embedding`.

  `DELETE`                `/delete/:id`                                        Delete a demo vector.

  `GET`                   `/items`                                             List demo vectors.

  `GET`                   `/benchmark?v=...&k=5&metric=cosine`                 Benchmark all three
                                                                               search algorithms.

  `GET`                   `/hnsw-info`                                         Return HNSW graph
                                                                               information and
                                                                               per-layer statistics.

  `GET`                   `/stats`                                             Return database
                                                                               statistics.

  `POST`                  `/doc/insert`                                        Insert a document using
                                                                               `title` and `text`;
                                                                               split, embed, and store
                                                                               it.

  `GET`                   `/doc/list`                                          List stored document
                                                                               chunks.

  `DELETE`                `/doc/delete/:id`                                    Delete a document
                                                                               chunk.

  `POST`                  `/doc/search`                                        Retrieve relevant
                                                                               chunks using `question`
                                                                               and `k`.

  `POST`                  `/doc/ask`                                           Run RAG: retrieve
                                                                               relevant chunks and
                                                                               generate an answer.

  `GET`                   `/status`                                            Return Ollama status
                                                                               and model information.
  ----------------------------------------------------------------------------------------------------

### Example requests

Search the demo vectors locally:

``` bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3"
```

Ask a question using the RAG endpoint (requires Ollama):

``` bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Deployment

The project is deployed on Render:

**https://vectormind-ai-nsg8.onrender.com/**

Suggested Render Web Service settings:

-   **Runtime:** Python
-   **Build command:** `pip install -r requirements.txt`
-   **Start command:** `python main.py`
-   **Root directory:** the directory containing `main.py`,
    `index.html`, and `requirements.txt`. Leave this blank if those
    files are at the repository root.

The server should bind to `0.0.0.0` and use the port provided by the
hosting platform through the `PORT` environment variable (with a local
fallback such as `8080`).

If the repository contains a nested project directory, set Render's
**Root Directory** to that exact directory relative to the repository
root.

## Project Structure

The application directory contains:

``` text
VectorMind-AI/
├── main.py            # Python backend, search algorithms, REST API, and RAG
├── index.html         # Browser-based frontend
├── requirements.txt   # Empty when using only the Python standard library
└── README.md
```

## Future Improvements

-   Configure a secure remote embedding and language-model service for
    cloud-hosted RAG.
-   Add persistent storage so inserted vectors and documents survive
    service restarts.
-   Add authentication and request validation before exposing
    write/delete API endpoints publicly.
-   Expand benchmark datasets and report measured latency and recall
    across different dataset sizes.

## License

This project is described as MIT-licensed in the original README.
Include a `LICENSE` file containing the MIT License text in the
repository if it is not already present.
