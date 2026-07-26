from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


_SPACE_RE = re.compile(r"\s+")


def normalize_name(value: object) -> str:
    return _SPACE_RE.sub(" ", "" if value is None else str(value).strip().lower())


@dataclass(frozen=True)
class PairRecord:
    drug: str
    disease: str
    label: int

    @property
    def key(self) -> tuple[str, str]:
        return self.drug, self.disease


@dataclass(frozen=True)
class PreparedPairs:
    train: tuple[PairRecord, ...]
    validation: tuple[PairRecord, ...]
    test: tuple[PairRecord, ...]
    audit: dict[str, object]


def _parse_label(raw: str, path: Path, line_number: int) -> int:
    try:
        label = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid label at {path}:{line_number}: {raw!r}") from exc
    if label not in {-1, 0, 1}:
        raise ValueError(f"Unsupported label at {path}:{line_number}: {label}")
    return label


def iter_pair_tsv(path: Path, *, allowed_labels: set[int] | None = None) -> Iterator[PairRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 1):
            if not row or all(not value.strip() for value in row):
                continue
            if line_number == 1 and len(row) >= 3 and row[2].strip().lower() in {"label", "output"}:
                continue
            if len(row) != 3:
                raise ValueError(f"Expected three TSV columns at {path}:{line_number}, found {len(row)}")
            drug = normalize_name(row[0])
            disease = normalize_name(row[1])
            if not drug or not disease:
                raise ValueError(f"Empty drug or disease at {path}:{line_number}")
            label = _parse_label(row[2], path, line_number)
            if allowed_labels is not None and label not in allowed_labels:
                raise ValueError(
                    f"Unexpected label {label} at {path}:{line_number}; expected {sorted(allowed_labels)}"
                )
            yield PairRecord(drug, disease, label)


def load_pair_tsv(path: Path, *, allowed_labels: set[int]) -> list[PairRecord]:
    records = list(iter_pair_tsv(path, allowed_labels=allowed_labels))
    seen: dict[tuple[str, str], int] = {}
    for record in records:
        previous = seen.get(record.key)
        if previous is not None:
            if previous != record.label:
                raise ValueError(f"Conflicting duplicate pair in {path}: {record.key}")
            raise ValueError(f"Duplicate pair in {path}: {record.key}")
        seen[record.key] = record.label
    return records


def load_holdout_keys(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 1):
            if not row or all(not value.strip() for value in row):
                continue
            if line_number == 1 and row[0].strip().lower() in {"drug", "head_name"}:
                continue
            if len(row) not in {2, 3}:
                raise ValueError(f"Expected two or three columns at {path}:{line_number}")
            key = normalize_name(row[0]), normalize_name(row[1])
            if not all(key):
                raise ValueError(f"Empty holdout pair at {path}:{line_number}")
            keys.add(key)
    return keys


def load_direct_kg_pairs(edges_dir: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for filename in ("Drug__TREATS__Disease.tsv", "Drug__CONTRAINDICATES__Disease.tsv"):
        path = edges_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Clean validation requires direct-edge audit file: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header is None or len(header) < 3:
                raise ValueError(f"Invalid KG edge file: {path}")
            for line_number, row in enumerate(reader, 2):
                if len(row) < 3:
                    raise ValueError(f"Invalid KG edge row at {path}:{line_number}")
                pairs.add((normalize_name(row[0]), normalize_name(row[2])))
    return pairs


def _stratified_clean_split(
    records: Sequence[PairRecord],
    direct_kg_pairs: set[tuple[str, str]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[PairRecord], list[PairRecord], dict[str, object]]:
    rng = random.Random(seed)
    clean_by_label: dict[int, list[PairRecord]] = defaultdict(list)
    direct_seen: list[PairRecord] = []
    for record in records:
        if record.key in direct_kg_pairs:
            direct_seen.append(record)
        else:
            clean_by_label[record.label].append(record)

    train: list[PairRecord] = []
    validation: list[PairRecord] = []
    for label in (0, 1):
        items = list(clean_by_label[label])
        rng.shuffle(items)
        n_validation = max(1, int(round(len(items) * validation_fraction))) if items else 0
        validation.extend(items[:n_validation])
        train.extend(items[n_validation:])
    train.extend(direct_seen)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation, {
        "direct_kg_pairs_forced_to_train": len(direct_seen),
        "direct_kg_label_counts": dict(Counter(record.label for record in direct_seen)),
    }


def prepare_supervised_pairs(
    positive_path: Path,
    negative_path: Path,
    test_path: Path,
    holdout_path: Path | None,
    edges_dir: Path,
    validation_fraction: float,
    seed: int,
) -> PreparedPairs:
    positive = load_pair_tsv(positive_path, allowed_labels={1})
    negative = load_pair_tsv(negative_path, allowed_labels={0})
    test = load_pair_tsv(test_path, allowed_labels={0, 1})

    positive_keys = {record.key for record in positive}
    negative_keys = {record.key for record in negative}
    conflicts = positive_keys & negative_keys
    if conflicts:
        example = next(iter(conflicts))
        raise ValueError(f"P/N label conflict for pair: {example}")

    test_keys = {record.key for record in test}
    holdout_keys = load_holdout_keys(holdout_path)
    excluded = test_keys | holdout_keys
    supervised = [record for record in positive + negative if record.key not in excluded]
    retained_keys = {record.key for record in supervised}
    if retained_keys & test_keys:
        raise RuntimeError("Test leakage remained after filtering")
    if retained_keys & holdout_keys:
        raise RuntimeError("Holdout leakage remained after filtering")

    direct_kg_pairs = load_direct_kg_pairs(edges_dir)
    train, validation, split_audit = _stratified_clean_split(
        supervised,
        direct_kg_pairs,
        validation_fraction,
        seed,
    )
    validation_direct_hits = sum(record.key in direct_kg_pairs for record in validation)
    if validation_direct_hits:
        raise RuntimeError("Clean validation contains direct TREATS/CONTRAINDICATES pairs")

    audit: dict[str, object] = {
        "seed": seed,
        "raw_positive": len(positive),
        "raw_negative": len(negative),
        "test": len(test),
        "test_positive": sum(record.label == 1 for record in test),
        "test_negative": sum(record.label == 0 for record in test),
        "test_overlap_removed_from_positive": len(positive_keys & test_keys),
        "test_overlap_removed_from_negative": len(negative_keys & test_keys),
        "holdout_pairs": len(holdout_keys),
        "holdout_overlap_removed_from_positive": len(positive_keys & holdout_keys),
        "holdout_overlap_removed_from_negative": len(negative_keys & holdout_keys),
        "supervised_after_exclusion": len(supervised),
        "train": len(train),
        "validation": len(validation),
        "train_label_counts": dict(Counter(record.label for record in train)),
        "validation_label_counts": dict(Counter(record.label for record in validation)),
        **split_audit,
    }
    return PreparedPairs(tuple(train), tuple(validation), tuple(test), audit)


def pair_entity_sets(records: Iterable[PairRecord]) -> tuple[set[str], set[str]]:
    drugs: set[str] = set()
    diseases: set[str] = set()
    for record in records:
        drugs.add(record.drug)
        diseases.add(record.disease)
    return drugs, diseases


def write_pairs(path: Path, records: Sequence[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for record in records:
            writer.writerow((record.drug, record.disease, record.label))
