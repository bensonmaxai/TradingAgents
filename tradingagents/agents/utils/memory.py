"""Financial situation memory using BM25 for lexical similarity matching.

Uses BM25 (Best Matching 25) algorithm for retrieval - no API calls,
no token limits, works offline with any LLM provider.
"""

from rank_bm25 import BM25Okapi
from typing import List, Tuple
from datetime import datetime, date
import re


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situations using BM25."""

    def __init__(self, name: str, config: dict = None, max_documents: int = 50):
        """Initialize the memory system.

        Args:
            name: Name identifier for this memory instance
            config: Configuration dict (kept for API compatibility, not used for BM25)
            max_documents: Maximum documents to retain (FIFO eviction). 0 = unlimited.
        """
        self.name = name
        self.max_documents = max_documents
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.pinned_documents: List[str] = []
        self.pinned_recommendations: List[str] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing.

        Simple whitespace + punctuation tokenization with lowercasing.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

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
        scores = self.bm25.get_scores(query_tokens)

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
        if not include_pinned and self.pinned_documents:
            self._rebuild_index()


if __name__ == "__main__":
    from datetime import timedelta

    print("=== Tiered Memory Test ===\n")
    matcher = FinancialSituationMemory("test_memory")
    today = date.today()

    # Pinned playbook
    matcher.pinned_documents = [
        "BTCUSDT volatility spike with leverage flush (pinned playbook)",
    ]
    matcher.pinned_recommendations = [
        "PINNED: Always use tight stops on leveraged crypto positions during high volatility.",
    ]

    # Time-varied memories (same topic, different ages)
    example_data = [
        (
            f"BTCUSDT dropped 5% after Fed hawkish comments ({(today - timedelta(days=60)).isoformat()})",
            "OLD (60d): Fed rhetoric caused panic but recovered within 2 weeks.",
        ),
        (
            f"BTCUSDT rallied 8% on ETF inflow news ({(today - timedelta(days=15)).isoformat()})",
            "MID (15d): ETF flows are reliable short-term catalyst for BTC.",
        ),
        (
            f"BTCUSDT whipsawed 3% on mixed CPI data ({(today - timedelta(days=3)).isoformat()})",
            "RECENT (3d): CPI surprises cause initial knee-jerk then reversal — wait 4h before acting.",
        ),
    ]
    matcher.add_situations(example_data)

    query = "BTCUSDT showing volatility after macro data release"

    print("--- Without time weighting ---")
    results = matcher.get_memories(query, n_matches=3)
    for i, rec in enumerate(results, 1):
        print(f"  #{i} score={rec['similarity_score']:.3f} | {rec['recommendation'][:80]}")

    print("\n--- With time weighting (reference_date=today) ---")
    results = matcher.get_memories(query, n_matches=3, reference_date=today)
    for i, rec in enumerate(results, 1):
        print(f"  #{i} score={rec['similarity_score']:.3f} | {rec['recommendation'][:80]}")

    print("\nExpected: Pinned > Recent(3d) > Mid(15d) > Old(60d)")
