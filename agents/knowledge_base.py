"""
knowledge_base.py — Simple Vector Store for RAG
===============================================
Uses LanceDB for efficient vector similarity search of evidence.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Vector store knowledge base using LanceDB and SentenceTransformers."""

    def __init__(self, db_path: Optional[Path] = None, table_name: Optional[str] = None):
        from config import KB_DIR, LANCEDB_TABLE_NAME, EMBEDDING_MODEL_NAME
        self.db_path = db_path or KB_DIR
        self.table_name = table_name or LANCEDB_TABLE_NAME
        self.model_name = EMBEDDING_MODEL_NAME
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db, self._table, self._encoder = None, None, None

    def _get_resources(self):
        """Lazy load DB and Encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(str(self.db_path))
        if self._table is None and self.table_name in self._db.table_names():
            self._table = self._db.open_table(self.table_name)
        return self._db, self._table, self._encoder

    def build_from_vifactcheck(self, jsonl_path: Optional[Path] = None) -> int:
        """Builds the KB from a JSONL file of evidence."""
        from config import VIFACTCHECK_DIR
        jsonl_path = jsonl_path or (VIFACTCHECK_DIR / "vifactcheck_kb.jsonl")
        if not jsonl_path.exists():
            logger.error(f"Source not found: {jsonl_path}")
            return 0

        entries = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8") if line.strip()]
        db, _, encoder = self._get_resources()
        
        texts = [f"{e.get('claim', '')} {e.get('evidence', '')}" for e in entries]
        logger.info(f"Encoding {len(texts)} entries...")
        embeddings = encoder.encode(texts, show_progress_bar=True)

        records = [{
            "claim": e.get("claim", ""),
            "evidence": e.get("evidence", ""),
            "label": e.get("label", ""),
            "vector": embeddings[i].tolist()
        } for i, e in enumerate(entries)]

        if self.table_name in db.table_names():
            db.drop_table(self.table_name)
        self._table = db.create_table(self.table_name, records)
        return len(records)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Vector similarity search."""
        _, table, encoder = self._get_resources()
        if table is None: return []
        
        res = table.search(encoder.encode(query)).limit(top_k).to_pandas()
        return [{
            "claim": row["claim"],
            "evidence": row["evidence"],
            "label": row["label"],
            "score": float(row["_distance"])
        } for _, row in res.iterrows()]
