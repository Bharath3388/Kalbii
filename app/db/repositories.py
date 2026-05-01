"""Mongo client + record repository.

Designed so tests can swap in `mongomock` by setting the env var
``KMI_USE_MONGOMOCK=1`` before importing this module.
"""

from __future__ import annotations
import datetime as dt
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _client():
    if os.getenv("KMI_USE_MONGOMOCK") == "1":
        import mongomock
        return mongomock.MongoClient()
    from pymongo import MongoClient
    return MongoClient(get_settings().MONGO_URI, serverSelectionTimeoutMS=5000)


def _coll():
    db = _client()[get_settings().MONGO_DB]
    coll = db["records"]
    try:
        coll.create_index("job_id", unique=True)
        coll.create_index([("created_at", -1)])
        coll.create_index("risk_label")
    except Exception:                                             # noqa: BLE001
        pass
    return coll


class RecordsRepo:
    def insert(self, doc: Dict[str, Any]) -> str:
        doc = {**doc}
        doc.setdefault("created_at", dt.datetime.utcnow())
        res = _coll().insert_one(doc)
        return str(res.inserted_id)

    def by_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        d = _coll().find_one({"job_id": job_id})
        if d:
            d["_id"] = str(d["_id"])
        return d

    def recent(self, limit: int = 50, risk_label: Optional[str] = None) -> List[Dict[str, Any]]:
        q: Dict[str, Any] = {}
        if risk_label:
            q["risk_label"] = risk_label.upper()
        cur = _coll().find(q).sort("created_at", -1).limit(limit)
        out = []
        for d in cur:
            d["_id"] = str(d["_id"])
            out.append(d)
        return out

    def count(self) -> int:
        return _coll().count_documents({})

    def ping(self) -> bool:
        try:
            _client().admin.command("ping")
            return True
        except Exception:                                          # noqa: BLE001
            return False
