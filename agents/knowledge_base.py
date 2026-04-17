"""
knowledge_base.py — Vector Store cho ViFactCheck Evidence (RAG)
================================================================
Dùng LanceDB để lưu trữ và search evidence trích từ ViFactCheck dataset.
Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim).

Usage:
    from agents.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb.build_from_vifactcheck()
    results = kb.search("Vaccine COVID gây tự kỷ", top_k=5)
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
