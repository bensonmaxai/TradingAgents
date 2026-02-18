"""Financial situation memory using hybrid BM25 + vector search.

BM25 for exact keyword matching + Ollama embeddings for semantic similarity.
Falls back to BM25-only if Ollama is unavailable.
"""

from rank_bm25 import BM25Okapi
from typing import List, Tuple
from datetime import datetime, date
import json as _json
import re
import sys
import urllib.request

import numpy as np

# Hybrid search defaults (overridable via config)
_OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
_OLLAMA_EMBED_MODEL = "nomic-embed-text"
_OLLAMA_EMBED_TIMEOUT = 30
_OLLAMA_KEEP_ALIVE = "10m"
_HYBRID_ALPHA = 0.6  # BM25 weight; (1 - alpha) = vector weight
_EMBED_MAX_CHARS = 4500  # nomic-embed-text 8192 token context ≈ 4500 chars safe limit


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situations using BM25."""

    def __init__(self, name: str, config: dict = None, max_documents: int = 50):
        """Initialize the memory system.

        Args:
            name: Name identifier for this memory instance
            config: Configuration dict (hybrid_search key enables vector search)
            max_documents: Maximum documents to retain (FIFO eviction). 0 = unlimited.
        """
        self.name = name
        self.max_documents = max_documents
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.pinned_documents: List[str] = []
        self.pinned_recommendations: List[str] = []
        self.bm25 = None
        # Hybrid search (BM25 + vector)
        self._doc_embeddings = None     # (N, dim) ndarray cache
        self._embeddings_dirty = True   # True = need recompute
        self._hybrid_enabled = (config or {}).get("hybrid_search", True)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing.

        Simple whitespace + punctuation tokenization with lowercasing.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    @staticmethod
    def _ollama_embed(texts: list, keep_alive: str = _OLLAMA_KEEP_ALIVE):
        """Embed texts via Ollama /api/embed. Returns (N, dim) ndarray or None."""
        try:
            # Pre-truncate to fit nomic-embed-text context window
            truncated = [t[:_EMBED_MAX_CHARS] for t in texts]
            payload = _json.dumps({
                "model": _OLLAMA_EMBED_MODEL,
                "input": truncated,
                "keep_alive": keep_alive,
            }).encode("utf-8")
            req = urllib.request.Request(
                _OLLAMA_EMBED_URL, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=_OLLAMA_EMBED_TIMEOUT) as resp:
                data = _json.loads(resp.read())
            embs = data.get("embeddings")
            if embs:
                return np.array(embs, dtype=np.float32)
        except Exception as e:
            print(f"[memory] Ollama embed failed: {e}, BM25-only fallback", file=sys.stderr)
        return None

    def _ensure_embeddings(self):
        """Lazily compute document embeddings. Only calls Ollama when dirty."""
        if not self._embeddings_dirty or not self._hybrid_enabled:
            return
        all_docs = self.pinned_documents + self.documents
        if not all_docs:
            self._doc_embeddings = None
            self._embeddings_dirty = False
            return
        self._doc_embeddings = self._ollama_embed(all_docs)
        self._embeddings_dirty = False

    def _extract_date(self, text: str):
        """Extract date from memory document text (YYYY-MM-DD format in parens)."""
        m = re.search(r'\((\d{4}-\d{2}-\d{2})\)', text[:500])
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        return None

    def _rebuild_index(self):
        """Rebuild the BM25 index after adding documents (pinned + regular)."""
        all_docs = self.pinned_documents + self.documents
        if all_docs:
            tokenized_docs = [self._tokenize(doc) for doc in all_docs]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None
        self._embeddings_dirty = True

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        """Add financial situations and their corresponding advice.

        Args:
            situations_and_advice: List of tuples (situation, recommendation)
        """
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)

        # FIFO eviction: remove oldest if cap exceeded
        if self.max_documents > 0 and len(self.documents) > self.max_documents:
            excess = len(self.documents) - self.max_documents
            self.documents = self.documents[excess:]
            self.recommendations = self.recommendations[excess:]

        # Rebuild BM25 index with new documents
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 1,
                     reference_date=None) -> List[dict]:
        """Find matching recommendations using BM25 similarity with time weighting.

        Searches across both pinned (playbook) and regular (reflection) entries.
        When reference_date is provided, applies time-based weight multipliers:
          - Pinned: ×4.0 (always highest priority)
          - 0-7 days: ×3.0 (recent lessons most relevant)
          - 8-30 days: ×2.0 (mid-term patterns)
          - 31+ days: ×1.0 (long-term, lowest priority)

        Args:
            current_situation: The current financial situation to match against
            n_matches: Number of top matches to return
            reference_date: Date to calculate memory age from (None = no time weighting)

        Returns:
            List of dicts with matched_situation, recommendation, and similarity_score
        """
        all_docs = self.pinned_documents + self.documents
        all_recs = self.pinned_recommendations + self.recommendations

        if not all_docs or self.bm25 is None:
            return []

        # Tokenize query
        query_tokens = self._tokenize(current_situation)

        # Get BM25 scores for all documents
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Hybrid: combine BM25 + vector scores
        use_hybrid = False
        if self._hybrid_enabled:
            self._ensure_embeddings()
            if self._doc_embeddings is not None:
                query_emb = self._ollama_embed([current_situation])
                if query_emb is not None:
                    q = query_emb[0]
                    q = q / (np.linalg.norm(q) or 1.0)
                    norms = np.linalg.norm(self._doc_embeddings, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    d = self._doc_embeddings / norms
                    vector_scores = d @ q  # cosine similarity
                    use_hybrid = True

        if use_hybrid:
            bm25_max = float(bm25_scores.max()) or 1.0
            bm25_norm = bm25_scores / bm25_max
            vec_min = float(vector_scores.min())
            vec_max = float(vector_scores.max())
            vec_range = (vec_max - vec_min) or 1.0
            vec_norm = (vector_scores - vec_min) / vec_range
            scores = _HYBRID_ALPHA * bm25_norm + (1 - _HYBRID_ALPHA) * vec_norm
        else:
            scores = bm25_scores

        # Apply time-based weighting (FinMem-style tiered memory)
        if reference_date:
            n_pinned = len(self.pinned_documents)
            for i in range(len(scores)):
                if i < n_pinned:
                    scores[i] *= 4.0  # Pinned playbook: highest priority
                else:
                    doc_date = self._extract_date(all_docs[i])
                    if doc_date:
                        age_days = (reference_date - doc_date).days
                        if age_days <= 7:
                            scores[i] *= 3.0  # Recent
                        elif age_days <= 30:
                            scores[i] *= 2.0  # Mid-term
                        # else: ×1.0 (long-term, no change)

        # Get top-n indices sorted by score (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_matches]

        # Build results
        max_score = max(scores) if max(scores) > 0 else 1  # Normalize scores
        results = []

        for idx in top_indices:
            # Normalize score to 0-1 range for consistency
            normalized_score = scores[idx] / max_score if max_score > 0 else 0
            results.append({
                "matched_situation": all_docs[idx],
                "recommendation": all_recs[idx],
                "similarity_score": normalized_score,
            })

        return results

    def clear(self, include_pinned=False):
        """Clear stored memories. Pinned entries preserved unless include_pinned=True."""
        self.documents = []
        self.recommendations = []
        if include_pinned:
            self.pinned_documents = []
            self.pinned_recommendations = []
        self.bm25 = None
        self._doc_embeddings = None
        self._embeddings_dirty = True
        if not include_pinned and self.pinned_documents:
            self._rebuild_index()


if __name__ == "__main__":
    from datetime import timedelta

    print("=== Hybrid Search Test ===\n")
    today = date.today()

    # --- Test 1: Semantic matching ---
    print("--- Test 1: Semantic matching (BM25-only vs Hybrid) ---")
    matcher = FinancialSituationMemory("test_memory")
    matcher.pinned_documents = [
        "BTCUSDT volatility spike with leverage flush (pinned playbook)",
    ]
    matcher.pinned_recommendations = [
        "PINNED: Always use tight stops on leveraged crypto positions.",
    ]
    example_data = [
        (
            f"BTCUSDT dropped 5% after Fed hawkish comments ({(today - timedelta(days=60)).isoformat()})",
            "OLD (60d): Fed rhetoric caused panic but recovered within 2 weeks.",
        ),
        (
            f"Severe market downturn across all assets following banking crisis ({(today - timedelta(days=10)).isoformat()})",
            "CRISIS (10d): Systemic risk events require cash preservation — reduce all exposure.",
        ),
        (
            f"BTCUSDT whipsawed 3% on mixed CPI data ({(today - timedelta(days=3)).isoformat()})",
            "RECENT (3d): CPI surprises cause initial knee-jerk then reversal.",
        ),
    ]
    matcher.add_situations(example_data)

    # "crash" has no keyword overlap with "downturn/crisis", but semantic similarity is high
    query = "BTCUSDT crash after macro economic shock"

    print("  BM25-only:")
    matcher._hybrid_enabled = False
    results = matcher.get_memories(query, n_matches=3, reference_date=today)
    for i, rec in enumerate(results, 1):
        print(f"    #{i} score={rec['similarity_score']:.3f} | {rec['recommendation'][:70]}")

    print("  Hybrid (BM25 + vector):")
    matcher._hybrid_enabled = True
    matcher._embeddings_dirty = True
    results = matcher.get_memories(query, n_matches=3, reference_date=today)
    for i, rec in enumerate(results, 1):
        print(f"    #{i} score={rec['similarity_score']:.3f} | {rec['recommendation'][:70]}")

    print("  Expected: Hybrid should rank 'banking crisis/downturn' higher\n")

    # --- Test 2: Fallback ---
    print("--- Test 2: Graceful fallback (bad URL) ---")
    import tradingagents.agents.utils.memory as _mem
    saved = _mem._OLLAMA_EMBED_URL
    _mem._OLLAMA_EMBED_URL = "http://127.0.0.1:99999/api/embed"
    matcher._embeddings_dirty = True
    matcher._hybrid_enabled = True
    results = matcher.get_memories(query, n_matches=2, reference_date=today)
    print(f"  Got {len(results)} results (BM25 fallback): {results[0]['recommendation'][:60]}...")
    _mem._OLLAMA_EMBED_URL = saved
    print("  OK — fallback works\n")
