"""Persistent, versioned review annotations and immutable analysis snapshots."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wb_common import TYPE_CLASSES

FIELDS = ["cut_id", "target_present", "types", "dominant_type", "attrs", "interference",
          "usable", "confidence", "memo", "meta_seen", "review_mode", "updated_at", "revision"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def csv_bytes(rows):
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        row = dict(row)
        for key in ("types", "attrs", "interference"):
            if isinstance(row.get(key), list):
                row[key] = ";".join(row[key])
        w.writerow(row)
    return ("\ufeff" + s.getvalue()).encode("utf-8")


def normalize(raw, manifest_row, complete=True):
    cut_id = str(raw.get("cut_id", ""))
    if cut_id != str(manifest_row["cut_id"]):
        raise ValueError("목록에 없는 샘플입니다.")
    a = {key: str(raw.get(key, "") or "") for key in FIELDS}
    for key in ("types", "attrs", "interference"):
        value = raw.get(key, [])
        if isinstance(value, str):
            value = value.split(";")
        if not isinstance(value, list):
            raise ValueError("형태와 표시 값은 목록이어야 합니다.")
        a[key] = list(dict.fromkeys(str(v) for v in value if v))
    if any(t not in TYPE_CLASSES + ["other"] for t in a["types"]):
        raise ValueError("알 수 없는 콜 형태입니다.")
    for key, choices in [("target_present", ["", "y", "n", "unsure"]),
                         ("usable", ["", "y", "n", "hold"]),
                         ("confidence", ["", "1", "2", "3"])]:
        if a[key] not in choices:
            raise ValueError(f"{key}: 잘못된 값입니다.")
    a["review_mode"] = "evaluation" if manifest_row.get("split") == "test" else "development"
    a["meta_seen"] = "y" if raw.get("meta_seen") in ("y", True, 1) else "n"
    if len(a["memo"]) > 10000:
        raise ValueError("메모는 10,000자 이내로 작성해 주세요.")
    if a["dominant_type"] not in TYPE_CLASSES + ["", "other", "none"]:
        raise ValueError("대표 형태가 올바르지 않습니다.")
    if a["dominant_type"] in TYPE_CLASSES + ["other"] and a["dominant_type"] not in a["types"]:
        raise ValueError("대표 형태는 선택한 콜 형태에 포함되어야 합니다.")
    if not a["dominant_type"] and len(a["types"]) == 1:
        a["dominant_type"] = a["types"][0]
    if complete:
        if not all(a[k] for k in ("target_present", "usable", "confidence")):
            raise ValueError("발성 유무, 사용 여부, 확신도를 선택해 주세요.")
        if a["target_present"] == "n":
            a["usable"] = "n"
        elif a["target_present"] == "unsure" or a["confidence"] == "1":
            if a["usable"] == "y":
                a["usable"] = "hold"
        if a["usable"] == "y" and a["target_present"] == "y" and not a["types"]:
            raise ValueError("들어 있는 콜 형태를 선택해 주세요. 판단이 어렵다면 보류를 선택합니다.")
    return a


class ReviewStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "reviews.sqlite3"
        with self.connect() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS annotations (cut_id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS drafts (cut_id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS history (revision INTEGER PRIMARY KEY AUTOINCREMENT,
                    cut_id TEXT NOT NULL, body TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS exposures (cut_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS draws (draw_id TEXT PRIMARY KEY, body TEXT NOT NULL);
            """)

    def connect(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.execute("PRAGMA busy_timeout=30000")
        return c

    def _exposed(self, c, cut_id):
        return c.execute("SELECT 1 FROM exposures WHERE cut_id=?", (cut_id,)).fetchone() is not None

    def get(self, cut_id):
        with self.connect() as c:
            row = c.execute("SELECT body FROM drafts WHERE cut_id=?", (cut_id,)).fetchone()
            draft = row is not None
            if not row:
                row = c.execute("SELECT body FROM annotations WHERE cut_id=?", (cut_id,)).fetchone()
            a = json.loads(row[0]) if row else {"cut_id": cut_id}
            a["meta_seen"] = "y" if self._exposed(c, cut_id) or a.get("meta_seen") == "y" else "n"
            a["is_draft"] = draft
            return a

    def all(self):
        with self.connect() as c:
            rows = [json.loads(r[0]) for r in c.execute("SELECT body FROM annotations ORDER BY cut_id")]
        return rows

    def expose(self, cut_id):
        with self.connect() as c:
            c.execute("INSERT OR IGNORE INTO exposures VALUES (?,?)", (cut_id, now()))

    def save_many(self, values, draft=False):
        table = "drafts" if draft else "annotations"
        saved = []
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            for value in values:
                a = dict(value)
                if a.get("meta_seen") == "y":
                    c.execute("INSERT OR IGNORE INTO exposures VALUES (?,?)", (a["cut_id"], now()))
                a["meta_seen"] = "y" if self._exposed(c, a["cut_id"]) else "n"
                a["updated_at"] = now()
                if not draft:
                    cur = c.execute("INSERT INTO history(cut_id,body,updated_at) VALUES (?,?,?)",
                                    (a["cut_id"], json.dumps(a, ensure_ascii=False), a["updated_at"]))
                    a["revision"] = cur.lastrowid
                    c.execute("UPDATE history SET body=? WHERE revision=?",
                              (json.dumps(a, ensure_ascii=False), a["revision"]))
                    c.execute("DELETE FROM drafts WHERE cut_id=?", (a["cut_id"],))
                c.execute(f"INSERT INTO {table} VALUES (?,?) ON CONFLICT(cut_id) DO UPDATE SET body=excluded.body",
                          (a["cut_id"], json.dumps(a, ensure_ascii=False)))
                saved.append(a)
        return saved

    def draw(self, draw_id, body):
        with self.connect() as c:
            c.execute("INSERT INTO draws VALUES (?,?)", (draw_id, json.dumps(body, ensure_ascii=False)))

    def snapshot(self):
        # The annotation rows and their revision belong to the same read transaction.
        with self.connect() as c:
            c.execute("BEGIN")
            revision = c.execute("SELECT COALESCE(MAX(revision),0) FROM history").fetchone()[0]
            rows = [json.loads(r[0]) for r in c.execute("SELECT body FROM annotations ORDER BY cut_id")]
        return revision, rows


def write_snapshot(core: pd.DataFrame, rows, revision, destination: Path):
    """Overlay only development labels; preserve original split and test labels."""
    destination.mkdir(parents=True, exist_ok=True)
    d = core.copy()
    d["label_origin"] = "weak_note"
    d["review_usable"] = ""
    d["review_revision"] = ""
    positions = {str(v): i for i, v in enumerate(d.cut_id)}
    stats = dict(revision=revision, reviewed=len(rows), corrected_dev=0, excluded_dev=0,
                 evaluation_saved=0, independent_usable=0, outside_core=0)
    independent = []
    for a in rows:
        i = positions.get(a["cut_id"])
        if i is None:
            stats["outside_core"] += 1
            continue
        if d.at[i, "split"] == "test":
            stats["evaluation_saved"] += 1
            if (a.get("usable") == "y" and a.get("target_present") == "y"
                    and a.get("confidence") in ("2", "3") and a.get("meta_seen") != "y"):
                independent.append(a)
            continue
        d.at[i, "review_usable"] = a["usable"]
        d.at[i, "review_revision"] = str(a.get("revision", ""))
        excluded = a["usable"] != "y" or a["target_present"] != "y" or a["confidence"] == "1"
        if excluded:
            for column in ("task_group3", "task_species", "task_type_single", "task_type_multi"):
                d.at[i, column] = ""
            d.at[i, "label_origin"] = "human_excluded"
            stats["excluded_dev"] += 1
            continue
        known = [t for t in TYPE_CLASSES if t in a["types"]]
        # Single-label evaluation is for genuinely single-type cuts, not the dominant type of a mixture.
        single = known[0] if len(a["types"]) == len(known) == 1 else ""
        d.at[i, "task_type_single"] = single
        d.at[i, "task_type_multi"] = ";".join(known)
        d.at[i, "type_single"] = single
        d.at[i, "type_weak"] = ";".join(known)
        d.at[i, "label_origin"] = "human_review"
        stats["corrected_dev"] += 1
    stats["independent_usable"] = len(independent)
    snapshot_bytes = csv_bytes(rows)
    stats["annotations_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    (destination / "annotations_snapshot.csv").write_bytes(snapshot_bytes)
    (destination / "independent_gold.csv").write_bytes(csv_bytes(independent))
    d.to_csv(destination / "reviewed_manifest.csv", index=False, encoding="utf-8-sig")
    (destination / "review_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
