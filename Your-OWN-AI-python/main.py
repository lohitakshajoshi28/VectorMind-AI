"""
VectorDB - a vector database built from scratch in Python (standard library only).

Python port of the original C++ project: HNSW, KD-Tree and Brute Force search,
three distance metrics, a 16D demo dataset, a document store with real Ollama
embeddings, and a RAG pipeline (embed -> retrieve -> generate).

Run:  python main.py      then open http://localhost:8080
"""

import heapq
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

sys.setrecursionlimit(10000)  # KD-Tree recursion on skewed insert orders

DIMS = 16  # demo vectors; doc embedding size comes from Ollama at runtime
PORT = 8080
HERE = os.path.dirname(os.path.abspath(__file__))

Vec = List[float]
Hit = Tuple[float, int]  # (distance, id)
DistFn = Callable[[Vec, Vec], float]


# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: Vec


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: Vec, b: Vec) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def cosine(a: Vec, b: Vec) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: Vec, b: Vec) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self) -> None:
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: Vec, k: int, dist: DistFn) -> List[Hit]:
        r = [(dist(q, v.emb), v.id) for v in self.items]
        r.sort()
        return r[:k]

    def remove(self, id_: int) -> None:
        self.items = [v for v in self.items if v.id != id_]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _ins(self, n: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        if n is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < n.item.emb[ax]:
            n.left = self._ins(n.left, v, d + 1)
        else:
            n.right = self._ins(n.right, v, d + 1)
        return n

    def _knn(self, n: Optional[KDNode], q: Vec, k: int, d: int,
             dist: DistFn, heap: list) -> None:
        # heap is a max-heap on (dist, id), stored as (-dist, -id)
        if n is None:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, -n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer, farther = (n.left, n.right) if diff < 0 else (n.right, n.left)
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def knn(self, q: Vec, k: int, dist: DistFn) -> List[Hit]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist, heap)
        return sorted((-nd, -nid) for nd, nid in heap)

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW - Hierarchical Navigable Small World
# =====================================================================

@dataclass
class _HNode:
    item: VectorItem
    max_lyr: int
    nbrs: List[List[int]] = field(default_factory=list)


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.G: Dict[int, _HNode] = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)

    def _rand_level(self) -> int:
        u = 1.0 - self.rng.random()  # (0, 1] so log() is safe
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: Vec, ep: int, ef: int, lyr: int,
                      dist: DistFn) -> List[Hit]:
        G = self.G
        vis = {ep}
        d0 = dist(q, G[ep].item.emb)
        cands = [(d0, ep)]            # min-heap: closest candidate first
        found = [(-d0, -ep)]          # max-heap on distance (negated)

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = G[cid]
            if lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in G:
                    continue
                vis.add(nid)
                nd = dist(q, G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, -nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted((-d, -i) for d, i in found)

    @staticmethod
    def _select_nbrs(cands: List[Hit], max_m: int) -> List[int]:
        return [i for _, i in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        G = self.G
        id_ = item.id
        lvl = self._rand_level()
        G[id_] = _HNode(item, lvl, [[] for _ in range(lvl + 1)])

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in G:
                    continue
                nn = G[nid].nbrs
                while len(nn) <= lc:
                    nn.append([])
                conn = nn[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = sorted((dist(G[nid].item.emb, G[c].item.emb), c)
                                for c in conn if c in G)
                    nn[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q: Vec, k: int, ef: int, dist: DistFn) -> List[Hit]:
        if self.entry_pt == -1:
            return []
        G = self.G
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_: int) -> None:
        if id_ not in self.G:
            return
        for nd in self.G.values():
            for i, layer in enumerate(nd.nbrs):
                if id_ in layer:
                    nd.nbrs[i] = [x for x in layer if x != id_]
        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id_:
                    self.entry_pt = nid
                    break
        del self.G[id_]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes, edges = [], []
        for id_, nd in self.G.items():
            nodes.append({"id": id_, "metadata": nd.item.metadata,
                          "category": nd.item.category, "maxLyr": nd.max_lyr})
            for lc in range(nd.max_lyr + 1):
                if lc >= max_l:
                    break
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {"topLayer": self.top_layer, "nodeCount": len(self.G),
                "nodesPerLayer": nodes_per_layer,
                "edgesPerLayer": edges_per_layer,
                "nodes": nodes, "edges": edges}

    def __len__(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.RLock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: Vec, dist: DistFn) -> int:
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, list(emb))
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: Vec, k: int, metric: str, algo: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1e6)
            hits = [(d, self.store[i]) for d, i in raw if i in self.store]
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: Vec, k: int, metric: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)

            def timed(fn) -> int:
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1e6)

            return {
                "bruteforceUs": timed(lambda: self.bf.knn(q, k, dfn)),
                "kdtreeUs": timed(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "itemCount": len(self.store),
            }

    def all(self) -> List[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
    return chunks


# =====================================================================
#  OLLAMA CLIENT - wraps the local Ollama REST API
#  Install:  https://ollama.com
#  Models:   ollama pull nomic-embed-text
#            ollama pull llama3.2
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def _post(self, path: str, payload: dict, timeout: float) -> Optional[dict]:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    return None
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def embed(self, text: str) -> Vec:
        """Returns [] if Ollama is not running or the model is missing."""
        res = self._post("/api/embeddings",
                         {"model": self.embed_model, "prompt": text}, 30)
        if not res:
            return []
        return [float(x) for x in res.get("embedding", [])]

    def generate(self, prompt: str) -> str:
        res = self._post("/api/generate",
                         {"model": self.gen_model, "prompt": prompt,
                          "stream": False}, 180)  # LLMs can be slow
        if res is None:
            return "ERROR: Ollama unavailable. Run: ollama serve"
        return res.get("response", "")


# =====================================================================
#  DOCUMENT DATABASE - HNSW over real Ollama embeddings
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: Vec


class DocumentDB:
    def __init__(self) -> None:
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()  # brute force fallback for small sets
        self.mu = threading.RLock()
        self.next_id = 1
        self.dims = 0  # determined from first inserted embedding

    def insert(self, title: str, text: str, emb: Vec) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: Vec, k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            raw = (self.bf.knn(q, k, cosine) if len(self.store) < 10
                   else self.hnsw.knn(q, k, 50, cosine))
            return [(d, self.store[i]) for d, i in raw
                    if i in self.store and d <= max_dist]

    def remove(self, id_: int) -> bool:
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    data = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in data:
        db.insert(meta, cat, emb, dist)


# =====================================================================
#  HELPERS
# =====================================================================

def parse_vec(s: str) -> Vec:
    v = []
    for t in s.split(","):
        try:
            v.append(float(t))
        except ValueError:
            pass
    return v


def parse_int(s: str, default: int) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def rnd(v: float, n: int) -> float:
    return round(v, n)


# =====================================================================
#  HTTP SERVER
# =====================================================================

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


class Handler(BaseHTTPRequestHandler):
    server_version = "VectorDB/1.0"

    # -- plumbing ------------------------------------------------------
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), status=status)

    def _body(self) -> dict:
        n = parse_int(self.headers.get("Content-Length"), 0)
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    # -- routing -------------------------------------------------------
    def do_OPTIONS(self):
        self._send(b"", status=204)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        routes = {
            "/search": self.h_search, "/items": self.h_items,
            "/benchmark": self.h_benchmark, "/hnsw-info": self.h_hnsw_info,
            "/doc/list": self.h_doc_list, "/status": self.h_status,
            "/stats": self.h_stats,
        }
        if u.path in routes:
            return routes[u.path](q)
        if u.path == "/":
            return self.h_index()
        self._send(b"", status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        routes = {"/insert": self.h_insert, "/doc/insert": self.h_doc_insert,
                  "/doc/search": self.h_doc_search, "/doc/ask": self.h_doc_ask}
        if path in routes:
            return routes[path](self._body())
        self._send(b"", status=404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        m = re.fullmatch(r"/delete/(\d+)", path)
        if m:
            return self._json({"ok": db.remove(int(m.group(1)))})
        m = re.fullmatch(r"/doc/delete/(\d+)", path)
        if m:
            return self._json({"ok": doc_db.remove(int(m.group(1)))})
        self._send(b"", status=404)

    # -- demo vector endpoints -----------------------------------------
    def h_search(self, q):
        v = parse_vec(q.get("v", ""))
        if len(v) != DIMS:
            return self._json({"error": f"need {DIMS}D vector"})
        k = parse_int(q.get("k"), 5)
        metric = q.get("metric") or "cosine"
        algo = q.get("algo") or "hnsw"
        out = db.search(v, k, metric, algo)
        self._json({
            "results": [{"id": h.id, "metadata": h.metadata, "category": h.category,
                         "distance": rnd(d, 6),
                         "embedding": [rnd(x, 4) for x in h.emb]}
                        for d, h in out["hits"]],
            "latencyUs": out["us"], "algo": out["algo"], "metric": out["metric"],
        })

    def h_insert(self, body):
        meta = body.get("metadata") or ""
        cat = body.get("category") or ""
        emb = body.get("embedding") or []
        try:
            emb = [float(x) for x in emb]
        except (TypeError, ValueError):
            emb = []
        if not meta or not emb or len(emb) != DIMS:
            return self._json({"error": "invalid body"})
        self._json({"id": db.insert(meta, cat, emb, get_dist_fn("cosine"))})

    def h_items(self, _q):
        self._json([{"id": v.id, "metadata": v.metadata, "category": v.category,
                     "embedding": [rnd(x, 4) for x in v.emb]} for v in db.all()])

    def h_benchmark(self, q):
        v = parse_vec(q.get("v", ""))
        if len(v) != DIMS:
            return self._json({"error": f"need {DIMS}D vector"})
        k = parse_int(q.get("k"), 5)
        self._json(db.benchmark(v, k, q.get("metric") or "cosine"))

    def h_hnsw_info(self, _q):
        self._json(db.hnsw_info())

    # -- document + RAG endpoints ---------------------------------------
    def h_doc_insert(self, body):
        title = body.get("title") or ""
        text = body.get("text") or ""
        if not title or not text:
            return self._json({"error": "need title and text"})
        chunks = chunk_text(text, 250, 30)
        ids = []
        for i, chunk in enumerate(chunks):
            emb = ollama.embed(chunk)
            if not emb:
                return self._json({"error": "Ollama unavailable. Install from https://ollama.com "
                                            "then run: ollama pull nomic-embed-text && "
                                            "ollama pull llama3.2"})
            chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
            ids.append(doc_db.insert(chunk_title, chunk, emb))
        self._json({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})

    def h_doc_list(self, _q):
        out = []
        for d in doc_db.all():
            preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
            out.append({"id": d.id, "title": d.title, "preview": preview,
                        "words": d.text.count(" ") + 1})
        self._json(out)

    def h_doc_search(self, body):
        question = body.get("question") or ""
        k = parse_int(body.get("k"), 3)
        if not question:
            return self._json({"error": "need question"})
        q_emb = ollama.embed(question)
        if not q_emb:
            return self._json({"error": "Ollama unavailable"})
        hits = doc_db.search(q_emb, k)
        self._json({"contexts": [{"id": d.id, "title": d.title, "distance": rnd(dist, 4)}
                                 for dist, d in hits]})

    def h_doc_ask(self, body):
        question = body.get("question") or ""
        k = parse_int(body.get("k"), 3)
        if not question:
            return self._json({"error": "need question"})

        # Step 1: embed the question
        q_emb = ollama.embed(question)
        if not q_emb:
            return self._json({"error": "Ollama unavailable"})

        # Step 2: retrieve top-k relevant chunks
        hits = doc_db.search(q_emb, k)

        # Step 3: build prompt
        ctx = "".join(f"[{i + 1}] {d.title}:\n{d.text}\n\n" for i, (_, d) in enumerate(hits))
        prompt = (
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
            "'the context doesn't mention'. Just answer the question naturally.\n\n"
            f"Context:\n{ctx}Question: {question}\n\nAnswer:"
        )

        # Step 4: generate answer
        answer = ollama.generate(prompt)

        # Step 5: return everything
        self._json({
            "answer": answer, "model": ollama.gen_model,
            "contexts": [{"id": d.id, "title": d.title, "text": d.text,
                          "distance": rnd(dist, 4)} for dist, d in hits],
            "docCount": doc_db.size(),
        })

    # -- status / stats / UI ------------------------------------------
    def h_status(self, _q):
        self._json({
            "ollamaAvailable": ollama.is_available(),
            "embedModel": ollama.embed_model, "genModel": ollama.gen_model,
            "docCount": doc_db.size(), "docDims": doc_db.get_dims(),
            "demoDims": DIMS, "demoCount": db.size(),
        })

    def h_stats(self, _q):
        self._json({"count": db.size(), "dims": DIMS,
                    "algorithms": ["bruteforce", "kdtree", "hnsw"],
                    "metrics": ["euclidean", "cosine", "manhattan"]})

    def h_index(self):
        try:
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")
        except OSError:
            self._send(b"", status=404)


def main() -> None:
    load_demo(db)
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print(f"http://localhost:{PORT}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama:", "ONLINE" if ollama_up else "OFFLINE (install from ollama.com)")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
