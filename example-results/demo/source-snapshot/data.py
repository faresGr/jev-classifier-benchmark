"""Frozen splits; all models receive the same truncated text."""
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = {
    "comp.graphics": "Computer graphics, image rendering, graphics software and formats.",
    "rec.autos": "Cars, driving, automotive maintenance and vehicle purchasing.",
    "rec.sport.baseball": "Baseball games, players, teams and statistics.",
    "sci.space": "Space exploration, astronomy, spacecraft and orbital missions.",
}


def digest(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def demo_rows():
    # Intentionally easy synthetic text. A pipeline check, never benchmark evidence.
    topics = {
        "comp.graphics": ["rendering a scene", "loading a texture", "converting an image", "drawing a polygon", "saving a bitmap", "animating a mesh"],
        "rec.autos": ["repairing the engine", "replacing brake pads", "buying a sedan", "checking tire pressure", "changing motor oil", "fixing a transmission"],
        "rec.sport.baseball": ["a pitcher striking out", "a batter hitting a homer", "the baseball league", "a stolen base", "the batting average", "a baseball playoff"],
        "sci.space": ["a spacecraft launch", "a satellite orbit", "a lunar mission", "a rocket engine", "an astronaut spacewalk", "a planetary probe"],
    }
    rows = []
    for label, subjects in topics.items():
        for i, subject in enumerate(subjects):
            for j, phrasing in enumerate(["I have a question about", "Can someone explain", "Here is a discussion of", "I am researching", "Please share advice about", "I read a report about"]):
                text = f"{phrasing} {subject}. This is the discussion from session {i * 6 + j}."
                rows.append({"text": text, "label": label, "source_split": "pool"})
    return rows


def validate_bundle(rows, labels):
    seen_ids, seen_text, groups = set(), set(), {}
    for r in rows:
        if not isinstance(r.get("text"), str) or not r["text"].strip():
            raise ValueError("Each record needs nonempty text")
        if r.get("label") not in labels or r.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Unknown label or split")
        if r.get("id") is None or str(r["id"]) in seen_ids:
            raise ValueError("Missing or duplicate record id")
        seen_ids.add(str(r["id"]))
        h = digest(r["text"])
        if h in seen_text:
            raise ValueError("Duplicate normalized text; deduplicate before import")
        seen_text.add(h)
        group = r.get("group_id")
        if group is not None:
            group = str(group)
            if group in groups and groups[group] != r["split"]:
                raise ValueError("group_id crosses splits")
            groups[group] = r["split"]
    for split in ("train", "validation", "test"):
        if {r["label"] for r in rows if r["split"] == split} != set(labels):
            raise ValueError(f"Every label must appear in {split}")


def prepare(config, destination, input_path=None, labels_path=None):
    dest = Path(destination)
    if dest.exists():
        raise ValueError(f"{dest} already exists; choose a new dataset directory")
    max_chars = config["max_chars"]
    dropped = 0
    if input_path:
        if not labels_path:
            raise ValueError("--labels is required for custom JSONL")
        labels = json.loads(Path(labels_path).read_text())
        rows = read_rows(input_path)
        for r in rows:
            r["text"] = r["text"][:max_chars]
        source = "custom"
    else:
        labels = LABELS.copy()
        source = config["dataset"]
        if source == "demo":
            raw = demo_rows()
        elif source == "newsgroups":
            from sklearn.datasets import fetch_20newsgroups
            raw = []
            for split in ("train", "test"):
                data = fetch_20newsgroups(subset=split, categories=list(labels),
                    remove=("headers", "footers", "quotes"), data_home=str(dest.parent / ".cache"), shuffle=False)
                raw.extend({"text": text, "label": data.target_names[int(y)], "source_split": split}
                           for text, y in zip(data.data, data.target))
        else:
            raise ValueError("dataset must be demo or newsgroups, or use --input")
        clean, seen = [], set()
        for r in raw:
            text = r["text"][:max_chars].strip()
            h = digest(text)
            if not text or h in seen:
                dropped += 1
                continue
            seen.add(h)
            clean.append({**r, "text": text, "id": h})
        rows = []
        rng = np.random.default_rng(config["split_seed"])
        for label in labels:
            pool = [r for r in clean if r["label"] == label and r["source_split"] != "test"]
            rng.shuffle(pool)
            nval, ntest = config["validation_per_class"], config["test_per_class"]
            if source == "demo":
                test, pool = pool[:ntest], pool[ntest:]
            else:
                test = [r for r in clean if r["label"] == label and r["source_split"] == "test"]
                rng.shuffle(test)
                if len(test) < ntest:
                    raise ValueError(f"Not enough test examples for {label}")
                test = test[:ntest]
            if len(pool) < nval + max(config["train_per_class"]):
                raise ValueError(f"Not enough train/validation examples for {label}")
            for split, subset in [("validation", pool[:nval]), ("train", pool[nval:]), ("test", test)]:
                rows.extend({"id": r["id"], "text": r["text"], "label": label, "split": split} for r in subset)
    if "__REQUEST_FAILED__" in labels or len(labels) < 2 or not all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items()):
        raise ValueError("labels must map at least two string labels to descriptions")
    validate_bundle(rows, labels)
    dest.mkdir(parents=True)
    write_rows(dest / "examples.jsonl", rows)
    dump(dest / "labels.json", labels)
    dump(dest / "metadata.json", {"source": source, "synthetic": source == "demo", "max_chars": max_chars,
         "split_seed": config["split_seed"], "preparation_config": config, "dropped_empty_or_duplicate": dropped,
         "labels_sha256": hashlib.sha256((dest / "labels.json").read_bytes()).hexdigest(),
         "examples_sha256": hashlib.sha256((dest / "examples.jsonl").read_bytes()).hexdigest()})
    return rows


def load_bundle(path):
    path = Path(path)
    rows, labels = read_rows(path / "examples.jsonl"), json.loads((path / "labels.json").read_text())
    meta = json.loads((path / "metadata.json").read_text())
    actual = hashlib.sha256((path / "examples.jsonl").read_bytes()).hexdigest()
    if actual != meta["examples_sha256"] or hashlib.sha256((path / "labels.json").read_bytes()).hexdigest() != meta["labels_sha256"]:
        raise ValueError("Dataset changed after preparation; prepare a new bundle")
    validate_bundle(rows, labels)
    return rows, labels, meta


def training_subset(rows, labels, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(labels):
        pool = sorted((r for r in rows if r["split"] == "train" and r["label"] == label), key=lambda r: str(r["id"]))
        if len(pool) < per_class:
            raise ValueError(f"Requested {per_class} training records for {label}, only {len(pool)} available")
        order = rng.permutation(len(pool))
        selected.extend(pool[i] for i in order[:per_class])
    return selected
