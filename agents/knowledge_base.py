"""
knowledge_base.py — Vector Store cho ViFactCheck Evidence (RAG)
================================================================
Dùng LanceDB để lưu trữ và search evidence trích từ ViFactCheck dataset.
Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim).

Features:
- Vector similarity search (LanceDB)
- Hybrid search (vector + BM25)
- Cross-encoder reranking

Usage:
    from agents.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb.build_from_vifactcheck()
    results = kb.search("Vaccine COVID gây tự kỷ", top_k=5)
    # Or use hybrid search:
    results = kb.search_hybrid("Vaccine COVID gây tự kỷ", top_k=5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Vector store knowledge base cho RAG evidence search.

    Sử dụng LanceDB để lưu evidence từ ViFactCheck + Wikipedia VN.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        table_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import KB_DIR, LANCEDB_TABLE_NAME, EMBEDDING_MODEL_NAME

        self.db_path = db_path or KB_DIR
        self.table_name = table_name or LANCEDB_TABLE_NAME
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL_NAME

        self.db_path.mkdir(parents=True, exist_ok=True)

        self._db = None
        self._table = None
        self._encoder = None

    def _get_encoder(self):
        """Lazy load sentence-transformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.embedding_model_name)
                logger.info(f"📐 Loaded encoder: {self.embedding_model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install: pip install sentence-transformers"
                )
        return self._encoder

    def _get_db(self):
        """Lazy load LanceDB."""
        if self._db is None:
            try:
                import lancedb

                self._db = lancedb.connect(str(self.db_path))
            except ImportError:
                raise ImportError("lancedb required. Install: pip install lancedb")
        return self._db

    def _get_table(self):
        """Get existing table or return None."""
        if self._table is None:
            db = self._get_db()
            if self.table_name in db.table_names():
                self._table = db.open_table(self.table_name)
        return self._table

    def build_from_vifactcheck(self, jsonl_path: Optional[Path] = None) -> int:
        """Build knowledge base từ ViFactCheck evidence JSONL.

        Args:
            jsonl_path: Path to vifactcheck_kb.jsonl.

        Returns:
            Number of entries added.
        """
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import VIFACTCHECK_DIR

        if jsonl_path is None:
            jsonl_path = VIFACTCHECK_DIR / "vifactcheck_kb.jsonl"

        if not jsonl_path.exists():
            logger.error(f"❌ Knowledge base source not found: {jsonl_path}")
            logger.info("💡 Chạy: python dataset/download_datasets.py trước")
            return 0

        logger.info(f"📚 Building knowledge base from {jsonl_path}...")

        # Load entries
        entries = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entries.append(entry)

        if not entries:
            logger.warning("⚠️ No entries found in JSONL file.")
            return 0

        logger.info(f"  📄 Loaded {len(entries)} entries")

        # Encode
        encoder = self._get_encoder()
        texts = [f"{e.get('claim', '')} {e.get('evidence', '')}" for e in entries]

        logger.info("  🔄 Encoding texts (this may take a few minutes)...")
        embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=64)

        # Create LanceDB records
        records = []
        for i, entry in enumerate(entries):
            records.append({
                "claim": entry.get("claim", ""),
                "evidence": entry.get("evidence", ""),
                "label": entry.get("label", ""),
                "vector": embeddings[i].tolist(),
            })

        # Save to LanceDB
        db = self._get_db()

        # Drop existing table if exists
        if self.table_name in db.table_names():
            db.drop_table(self.table_name)

        self._table = db.create_table(self.table_name, records)

        logger.info(f"✅ Knowledge base built: {len(records)} entries in table '{self.table_name}'")
        return len(records)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Tìm evidence liên quan đến query.

        Args:
            query: Câu hỏi / claim cần tìm evidence.
            top_k: Số kết quả trả về.

        Returns:
            List of dicts: {claim, evidence, label, score}
        """
        table = self._get_table()
        if table is None:
            logger.warning("⚠️ Knowledge base chưa build. Chạy build_from_vifactcheck() trước.")
            return []

        encoder = self._get_encoder()
        query_embedding = encoder.encode(query)

        results = table.search(query_embedding).limit(top_k).to_pandas()

        output = []
        for _, row in results.iterrows():
            output.append({
                "claim": row.get("claim", ""),
                "evidence": row.get("evidence", ""),
                "label": row.get("label", ""),
                "score": float(row.get("_distance", 0.0)),
            })

        return output

    def search_multiple(self, queries: list[str], top_k: int = 3) -> dict[str, list[dict]]:
        """Search evidence cho nhiều queries.

        Args:
            queries: List of claim strings.
            top_k: Số kết quả per query.

        Returns:
            Dict: {query: [results]}
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k=top_k)
        return results

    @property
    def size(self) -> int:
        """Số entries trong knowledge base."""
        table = self._get_table()
        if table is None:
            return 0
        return len(table)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict]:
        """Hybrid search kết hợp vector similarity + BM25 keyword search.

        Args:
            query: Search query.
            top_k: Số kết quả trả về.
            alpha: Weight for vector search (0.5 = equal weight).

        Returns:
            List of dicts: {claim, evidence, label, score}
        """
        table = self._get_table()
        if table is None:
            logger.warning("⚠️ Knowledge base chưa build. Chạy build_from_vifactcheck() trước.")
            return []

        # Vector search
        vector_results = self.search(query, top_k=top_k * 2)

        # BM25 keyword search
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        # RRF fusion
        fused = self._rrf_fusion(vector_results, bm25_results, top_k, alpha=alpha)

        logger.info(f"  🔀 Hybrid: {len(fused)} results (vector={len(vector_results)}, bm25={len(bm25_results)})")
        return fused

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 keyword search fallback khi không có full-text index.

        Args:
            query: Search query.
            top_k: Số kết quả.

        Returns:
            List of dicts: {claim, evidence, label, score}
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("⚠️ rank-bm25 not installed. Run: pip install rank-bm25")
            return []

        table = self._get_table()
        if table is None:
            return []

        # Get all entries
        try:
            df = table.to_pandas()
        except Exception:
            return []

        if df.empty:
            return []

        # Tokenize
        corpus = [self._tokenize(str(row.get("evidence", ""))) for _, row in df.iterrows()]
        bm25 = BM25Okapi(corpus)

        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # Get top results
        results = []
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        for idx in top_indices:
            if scores[idx] > 0:
                row = df.iloc[idx]
                results.append({
                    "claim": row.get("claim", ""),
                    "evidence": row.get("evidence", ""),
                    "label": row.get("label", ""),
                    "score": float(scores[idx]),
                })

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        import re
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return [t for t in tokens if len(t) > 1]

    def _rrf_fusion(
        self,
        list_a: list[dict],
        list_b: list[dict],
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) for combining two result sets.

        Args:
            list_a: First result list (vector search).
            list_b: Second result list (BM25 search).
            top_k: Number of results to return.
            alpha: Weight factor for list_a.

        Returns:
            Fused list of dicts.
        """
        if not list_a and not list_b:
            return []
        if not list_a:
            return list_b[:top_k]
        if not list_b:
            return list_a[:top_k]

        # Build score maps
        score_a = {self._result_key(r): alpha * (1.0 / (rank + 1)) for rank, r in enumerate(list_a)}
        score_b = {(1 - alpha) * (1.0 / (rank + 1)) for rank, r in enumerate(list_b)}

        # Combine scores
        all_keys = set(score_a.keys()) | set(score_b.keys())
        fused_scores = {}
        for key in all_keys:
            fused_scores[key] = score_a.get(key, 0) + score_b.get(key, 0)

        # Sort and return top_k
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        result_map = {self._result_key(r): r for r in list_a + list_b}

        return [result_map[k] for k in sorted_keys[:top_k] if k in result_map]

    def _result_key(self, result: dict) -> str:
        """Generate unique key for a result dict."""
        return f"{result.get('claim', '')[:50]}|{result.get('evidence', '')[:50]}"

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank candidates using cross-encoder model.

        Args:
            query: The original query.
            candidates: List of candidate result dicts.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of dicts.
        """
        if not candidates:
            return []

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("⚠️ sentence-transformers required for reranking.")
            return candidates[:top_k]

        try:
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as exc:
            logger.warning(f"⚠️ Could not load cross-encoder: {exc}")
            return candidates[:top_k]

        # Create query-document pairs
        pairs = [(query, str(c.get("evidence", ""))) for c in candidates]
        scores = cross_encoder.predict(pairs)

        # Sort candidates by score
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k
        return [c for _, c in scored[:top_k]]


def main() -> None:
    """Build knowledge base từ ViFactCheck."""
    kb = KnowledgeBase()
    count = kb.build_from_vifactcheck()

    if count > 0:
        # Test search
        logger.info("\n🔍 Testing search...")
        test_queries = [
            "Vaccine COVID-19 có gây ra tác dụng phụ nguy hiểm không?",
            "Việt Nam đạt tăng trưởng GDP cao nhất khu vực",
        ]
        for query in test_queries:
            results = kb.search(query, top_k=3)
            logger.info(f"\n  Query: {query[:60]}...")
            for r in results:
                logger.info(f"    [{r['label']:>10s}] {r['evidence'][:80]}...")


if __name__ == "__main__":
    main()
