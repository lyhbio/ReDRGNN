#!/usr/bin/env python3
"""Extract the small release inputs that replace private source JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SPACE_RE = re.compile(r"\s+")
PAIR_RE = re.compile(r"\s*Drug:\s*(.*?);\s*Disease:\s*(.*?)\s*$", re.IGNORECASE)


def normalize_name(value: object) -> str:
    return SPACE_RE.sub(" ", "" if value is None else str(value).strip().lower())


def build_quality_metadata(source: Path, output: Path) -> int:
    metadata: dict[tuple[str, str], tuple[object, ...]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            entity_type = normalize_name(record.get("entity_type", "")).upper()
            entity = normalize_name(record.get("entity", ""))
            if entity_type not in {"DRUG", "DISEASE"} or not entity:
                continue
            description = (
                record.get("final_description")
                or record.get("enriched_description")
                or ""
            )
            normalized_description = normalize_name(description)
            empty = normalized_description in {"", "none", "null", "nan"}
            words = 0 if empty else len(normalized_description.split())
            metadata[(entity_type, entity)] = (
                entity_type,
                entity,
                int(bool(record.get("needs_review", False))),
                str(record.get("description_source", "")),
                words,
                int(empty),
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "entity_type",
                "entity",
                "needs_review",
                "description_source",
                "description_words",
                "empty_description",
            )
        )
        writer.writerows(metadata[key] for key in sorted(metadata))
    return len(metadata)


def build_authority_holdout(source: Path, output: Path) -> int:
    rows: dict[tuple[str, str], tuple[str, str, int]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            match = PAIR_RE.fullmatch(str(record.get("input", "")))
            if match is None:
                raise ValueError(f"Cannot parse authority pair at {source}:{line_number}")
            drug = normalize_name(match.group(1))
            disease = normalize_name(match.group(2))
            answer = normalize_name(record.get("output", ""))
            if answer not in {"yes", "no"}:
                raise ValueError(f"Invalid authority answer at {source}:{line_number}: {answer!r}")
            row = (drug, disease, int(answer == "yes"))
            previous = rows.get((drug, disease))
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting authority pair at {source}:{line_number}")
            rows[(drug, disease)] = row

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows.values())
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--quality-output", type=Path, required=True)
    parser.add_argument("--authority-output", type=Path, required=True)
    args = parser.parse_args()
    quality_count = build_quality_metadata(args.descriptions, args.quality_output)
    authority_count = build_authority_holdout(args.authority, args.authority_output)
    print(f"quality_rows={quality_count}")
    print(f"authority_rows={authority_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
