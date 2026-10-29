"""Evidence ledger: every analysis result becomes a content-addressed record.

Design borrowed from REA (morluto/rea):
- `evidence_id` = sha256 over the canonical identity (operation, parameters,
  subject identity, analysis_profile, normalized_result, coverage, limitations).
- `analysis_profile` is part of the identity, so a result produced under older
  hook semantics never masquerades as a current one.
- `authority` says what kind of claim the record makes:
    observation  - measured under an intervention (patching, ablation, next-token)
    inference    - a heuristic readout (logit lens, attention weights, SAE features)
    derived      - exact arithmetic on retained facts (DLA, comparisons, views)
    declared     - read from files without execution (static artifact inspection)
- `coverage` is producer-owned ({status: complete|partial|unknown, ...}); a
  missing fact is never "proven absent". `limitations` only explain.
- Snapshot cache: same subject + operation + parameters + profile -> same record,
  so an expensive patching grid is never recomputed.
- Views page into a retained record instead of resending it.
- `provenance.untrusted_content` marks results carrying text produced by the
  analysed model. That text is data, never an instruction to the agent.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Iterable

AUTHORITIES = ("observation", "inference", "derived", "declared", "unknown")
BUNDLE_FORMAT = "rellm-evidence-bundle/2"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


UNTRUSTED = {"untrusted_content": True, "source": "analysis_target"}
TRUSTED = {"untrusted_content": False, "source": "session_state"}


class EvidenceError(ValueError):
    def __init__(self, code: str, message: str, remediation: str = "") -> None:
        super().__init__(message)
        self.code, self.remediation = code, remediation

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), "remediation": self.remediation}}


class EvidenceStore:
    def __init__(self, autosave_path: str | None = None) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._cache: dict[str, str] = {}          # cache key -> evidence_id
        self._unknowns: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.autosave_path = autosave_path

    # ----------------------------------------------------------------- records
    @staticmethod
    def cache_key(operation: str, parameters: dict[str, Any], subject: dict[str, Any], analysis_profile: dict[str, Any] | None) -> str:
        return sha256_of({"operation": operation, "parameters": parameters, "subject": subject, "analysis_profile": analysis_profile})

    def lookup(self, operation: str, parameters: dict[str, Any], subject: dict[str, Any], analysis_profile: dict[str, Any] | None) -> dict[str, Any] | None:
        with self._lock:
            eid = self._cache.get(self.cache_key(operation, parameters, subject, analysis_profile))
            return dict(self._records[eid]) if eid else None

    def record(
        self,
        operation: str,
        parameters: dict[str, Any],
        subject: dict[str, Any],
        normalized_result: dict[str, Any],
        coverage: dict[str, Any] | None = None,
        limitations: Iterable[str] = (),
        authority: str = "observation",
        analysis_profile: dict[str, Any] | None = None,
        facet_availability: dict[str, Any] | None = None,
        untrusted: bool = True,
        links: Iterable[str] = (),
        cacheable: bool = True,
    ) -> dict[str, Any]:
        if authority not in AUTHORITIES:
            raise EvidenceError("bad_authority", f"authority must be one of {AUTHORITIES}")
        coverage = coverage or {"status": "complete", "scope": operation}
        if coverage.get("status") not in ("complete", "partial", "unknown"):
            raise EvidenceError("bad_coverage", "coverage.status must be complete|partial|unknown")
        identity = {
            "operation": operation,
            "parameters": parameters,
            "subject": subject,
            "analysis_profile": analysis_profile,
            "normalized_result": normalized_result,
            "coverage": coverage,
            "limitations": list(limitations),
        }
        evidence_id = "ev_" + sha256_of(identity)
        rec = {
            "evidence_id": evidence_id,
            **identity,
            "authority": authority,
            "facet_availability": facet_availability or {},
            "evidence_links": list(links),
            "provenance": UNTRUSTED if untrusted else TRUSTED,
            "created_at": _now(),
        }
        with self._lock:
            if evidence_id not in self._records:
                self._order.append(evidence_id)
                self._records[evidence_id] = rec
            if cacheable and coverage.get("status") == "complete":
                self._cache[self.cache_key(operation, parameters, subject, analysis_profile)] = evidence_id
        self._autosave(rec)
        return dict(rec)

    def get(self, evidence_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(evidence_id)
        if rec is None:
            raise EvidenceError("missing", f"no evidence {evidence_id} in this session",
                                "Use rellm_list_evidence to discover ids, or import a bundle with rellm_import_evidence_bundle.")
        return dict(rec)

    def summary(self, operation: str | None = None, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            ids = [i for i in self._order if operation is None or self._records[i]["operation"] == operation]
            page = ids[offset: offset + limit]
            rows = [{k: self._records[i][k] for k in ("evidence_id", "operation", "authority", "coverage", "created_at", "parameters")}
                    | {"subject": {k: v for k, v in self._records[i]["subject"].items() if k in ("model_id", "weights_sha256", "config_sha256")}}
                    for i in page]
        return {"total": len(ids), "offset": offset, "next_offset": offset + limit if offset + limit < len(ids) else None, "records": rows}

    # ------------------------------------------------------------------- views
    def view(self, evidence_id: str, collection: str, offset: int = 0, limit: int = 64,
             fields: list[str] | None = None) -> dict[str, Any]:
        """Page into a list-valued field of a retained record. `collection` is a dotted path inside normalized_result."""
        rec = self.get(evidence_id)
        node: Any = rec["normalized_result"]
        for part in collection.split("."):
            if not isinstance(node, dict) or part not in node:
                keys = sorted(k for k, v in (node.items() if isinstance(node, dict) else []) if isinstance(v, list))
                raise EvidenceError("bad_collection", f"{collection!r} not found in {evidence_id}", f"list-valued collections here: {keys}")
            node = node[part]
        if not isinstance(node, list):
            raise EvidenceError("bad_collection", f"{collection!r} is not a list")
        page = node[offset: offset + limit]
        if fields:
            page = [{k: r.get(k) for k in fields} if isinstance(r, dict) else r for r in page]
        return {"parent_evidence_id": evidence_id, "collection": collection, "total": len(node), "offset": offset,
                "limit": limit, "next_offset": offset + limit if offset + limit < len(node) else None,
                "parent_coverage": rec["coverage"], "rows": page}

    # ---------------------------------------------------------------- unknowns
    def record_unknown(self, question: str, domain: str, context_evidence_id: str | None = None, notes: str = "") -> dict[str, Any]:
        body = {"question": question, "domain": domain, "context_evidence_id": context_evidence_id, "notes": notes}
        unknown_id = "unk_" + sha256_of(body)[:24]
        rec = {"unknown_id": unknown_id, **body, "status": "open", "resolution": None, "resolved_by": None, "created_at": _now()}
        with self._lock:
            self._unknowns.setdefault(unknown_id, rec)
            return dict(self._unknowns[unknown_id])

    def update_unknown(self, unknown_id: str, status: str, resolution: str | None = None, resolved_by: str | None = None) -> dict[str, Any]:
        if status not in ("open", "resolved", "abandoned"):
            raise EvidenceError("bad_status", "status must be open|resolved|abandoned")
        with self._lock:
            rec = self._unknowns.get(unknown_id)
            if rec is None:
                raise EvidenceError("missing", f"no unknown {unknown_id}")
            if status == "resolved" and resolved_by and resolved_by not in self._records:
                raise EvidenceError("missing", f"resolved_by {resolved_by} is not a retained evidence id")
            rec.update(status=status, resolution=resolution, resolved_by=resolved_by)
            return dict(rec)

    def list_unknowns(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(u) for u in self._unknowns.values() if status is None or u["status"] == status]

    # ------------------------------------------------------------ bundle i/o
    def export_bundle(self, path: str) -> dict[str, Any]:
        with self._lock:
            records = [self._records[i] for i in self._order]
            unknowns = list(self._unknowns.values())
        bundle = {"format": BUNDLE_FORMAT, "exported_at": _now(), "records": records, "unknowns": unknowns}
        bundle["manifest_sha256"] = sha256_of({"records": [r["evidence_id"] for r in records], "unknowns": [u["unknown_id"] for u in unknowns]})
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, path)
        return {"path": os.path.abspath(path), "records": len(records), "unknowns": len(unknowns),
                "manifest_sha256": bundle["manifest_sha256"], "bytes": os.path.getsize(path)}

    def import_bundle(self, path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8") as fh:
            bundle = json.load(fh)
        if bundle.get("format") != BUNDLE_FORMAT:
            raise EvidenceError("bad_bundle", f"unsupported bundle format {bundle.get('format')!r}; expected {BUNDLE_FORMAT}")
        added, rejected = 0, []
        for rec in bundle.get("records", []):
            identity = {k: rec.get(k) for k in ("operation", "parameters", "subject", "analysis_profile", "normalized_result", "coverage", "limitations")}
            if rec.get("evidence_id") != "ev_" + sha256_of(identity):
                rejected.append({"evidence_id": rec.get("evidence_id"), "reason": "identity digest mismatch"})
                continue
            with self._lock:
                if rec["evidence_id"] not in self._records:
                    self._records[rec["evidence_id"]] = rec
                    self._order.append(rec["evidence_id"])
                    added += 1
        for u in bundle.get("unknowns", []):
            with self._lock:
                self._unknowns.setdefault(u["unknown_id"], u)
        return {"path": os.path.abspath(path), "imported": added, "rejected": rejected, "unknowns": len(bundle.get("unknowns", []))}

    def _autosave(self, rec: dict[str, Any]) -> None:
        if not self.autosave_path:
            return
        try:
            with open(self.autosave_path, "a", encoding="utf-8") as fh:
                fh.write(canonical_json(rec) + "\n")
        except OSError:
            pass
