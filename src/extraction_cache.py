"""Versioned, success-only cache for expensive LLM card extraction.

The cache deliberately lives outside the extraction engine.  A session digest
captures the complete cleaned session, while a contract digest captures every
input that can change the meaning or shape of an extraction.  Consequently a
prompt/model/schema change cannot silently reuse an older result.

Cache writes are atomic and only successful :class:`ExtractedPIU` values are
persisted.  A malformed, stale, partial, or failed entry is a cache miss; it is
never converted into a synthetic result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from pydantic import BaseModel

from src.models import ExtractedPIU


logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "extraction-cache/v1"


def _jsonable(value: Any) -> Any:
    """Convert supported model/config values into deterministic JSON values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    """Serialize a value with stable key ordering and no lossy coercion."""
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"无法为缓存契约序列化值: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def compute_session_hash(session: Any) -> str:
    """Return a digest of the complete cleaned session payload.

    The function intentionally hashes the entire object, rather than only the
    intervention ID or dialogue text.  Question metadata, taxonomy, counts,
    and turn annotations are all part of the extraction input contract.
    """
    return _sha256({"session": _jsonable(session)})


def compute_contract_hash(
    contract: Any = None,
    *,
    prompt_version: Optional[str] = None,
    schema_version: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
    semantic_only: Optional[bool] = None,
    **extra: Any,
) -> str:
    """Return a digest of the complete extraction contract.

    ``contract`` is normally a mapping containing prompt/schema/model settings.
    Keyword fields are provided for callers that build the contract from
    command-line options; when supplied they are merged into that mapping.
    Unknown keyword fields are retained so adding a future extraction option
    automatically invalidates old cache entries.
    """
    if contract is None:
        contract_payload: dict[str, Any] = {}
    else:
        converted = _jsonable(contract)
        if isinstance(converted, Mapping):
            contract_payload = dict(converted)
        else:
            contract_payload = {"contract": converted}

    named_fields = {
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "model": model,
        "temperature": temperature,
        "max_retries": max_retries,
        "semantic_only": semantic_only,
    }
    contract_payload.update({key: value for key, value in named_fields.items() if value is not None})
    contract_payload.update(extra)
    return _sha256({"contract": contract_payload})


# Short aliases make the cache key useful to callers that use the term
# "fingerprint" in manifests.
session_fingerprint = compute_session_hash
contract_fingerprint = compute_contract_hash


def build_cache_key(session: Any, contract: Any = None, **contract_kwargs: Any) -> Tuple[str, str, str]:
    """Return ``(session_hash, contract_hash, entry_key)`` for a cache lookup."""
    session_hash = compute_session_hash(session)
    contract_hash = compute_contract_hash(contract, **contract_kwargs)
    return session_hash, contract_hash, f"{session_hash}:{contract_hash}"


class ExtractionCache:
    """A small JSON cache with atomic writes and strict success-only reads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("抽取缓存不可读，按 cache miss 处理 (%s): %s", self.path, exc)
            return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        if not isinstance(document, dict) or document.get("schema_version") != CACHE_SCHEMA_VERSION:
            logger.warning("抽取缓存 schema 不匹配，按 cache miss 处理: %s", self.path)
            return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        entries = document.get("entries")
        if not isinstance(entries, dict):
            logger.warning("抽取缓存 entries 非对象，按 cache miss 处理: %s", self.path)
            return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": entries}

    def get(self, session: Any, contract: Any = None, **contract_kwargs: Any) -> Optional[ExtractedPIU]:
        """Load a valid successful extraction or return ``None`` on any miss."""
        session_hash, contract_hash, entry_key = build_cache_key(session, contract, **contract_kwargs)
        entry = self._read_document()["entries"].get(entry_key)
        if not isinstance(entry, Mapping):
            return None
        if entry.get("session_hash") != session_hash or entry.get("contract_hash") != contract_hash:
            return None
        if entry.get("status") != "success":
            return None
        payload = entry.get("result")
        if not isinstance(payload, Mapping):
            return None
        try:
            result = ExtractedPIU.model_validate(payload)
        except Exception as exc:
            logger.warning("抽取缓存条目校验失败，按 cache miss 处理 (%s): %s", entry_key, exc)
            return None
        if (
            result.extraction_status != "success"
            or result.misconception is None
            or result.tutor_strategy is None
        ):
            return None
        return result

    def put(self, session: Any, contract: Any, result: ExtractedPIU, **contract_kwargs: Any) -> bool:
        """Persist a successful result atomically; return ``False`` otherwise."""
        if not isinstance(result, ExtractedPIU):
            raise TypeError("抽取缓存只接受 ExtractedPIU 实例")
        if (
            result.extraction_status != "success"
            or result.misconception is None
            or result.tutor_strategy is None
        ):
            logger.warning(
                "Session %s 抽取结果不是完整 success，不写入缓存",
                result.session_id,
            )
            return False

        session_hash, contract_hash, entry_key = build_cache_key(session, contract, **contract_kwargs)
        document = self._read_document()
        document["entries"][entry_key] = {
            "session_hash": session_hash,
            "contract_hash": contract_hash,
            "status": "success",
            "result": result.model_dump(mode="json"),
        }
        self._atomic_write(document)
        return True

    # ``set`` is a conventional cache spelling and intentionally shares the
    # strict put behavior.
    set = put

    def put_failure(
        self,
        session: Any,
        contract: Any,
        error_type: str,
        error_message: str,
    ) -> None:
        """Record an explicit failure without making it cacheable.

        Failure entries preserve resumability/audit history, while ``get``
        continues to treat them as misses so a later run can retry them.
        """
        session_hash, contract_hash, entry_key = build_cache_key(session, contract)
        document = self._read_document()
        document["entries"][entry_key] = {
            "session_hash": session_hash,
            "contract_hash": contract_hash,
            "status": "failed",
            "error_type": str(error_type),
            "error_message": str(error_message),
        }
        self._atomic_write(document)

    def _atomic_write(self, document: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_name = handle.name
                json.dump(document, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            temp_name = None
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


def load_cached_extraction(
    cache_path: str | Path,
    session: Any,
    contract: Any = None,
    **contract_kwargs: Any,
) -> Optional[ExtractedPIU]:
    """Functional convenience wrapper around :class:`ExtractionCache.get`."""
    return ExtractionCache(cache_path).get(session, contract, **contract_kwargs)


def save_cached_extraction(
    cache_path: str | Path,
    session: Any,
    contract: Any,
    result: ExtractedPIU,
    **contract_kwargs: Any,
) -> bool:
    """Functional convenience wrapper around :class:`ExtractionCache.put`."""
    return ExtractionCache(cache_path).put(session, contract, result, **contract_kwargs)
