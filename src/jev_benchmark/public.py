"""Well-known public text-classification datasets, fetched from the Hugging Face Hub.

Each spec maps the source dataset's own label names to the natural-language
descriptions Jev receives. The Hub commit that was actually downloaded is
resolved first and recorded, so a prepared bundle can be traced to exact data.
Loading needs the optional dependency: ``pip install -e '.[public]'``.
"""
import re

AG_NEWS = {
    "World": "International news: world politics, governments, conflicts, diplomacy and elections.",
    "Sports": "Sports news: games, matches, athletes, teams, leagues and tournaments.",
    "Business": "Business and economic news: companies, earnings, markets, oil prices, trade and finance.",
    "Sci/Tech": "Science and technology news: computing, software, the internet, telecoms, space and research.",
}

DBPEDIA_14 = {
    "Company": "A company, business, corporation or brand.",
    "EducationalInstitution": "A school, college, university or other educational institution.",
    "Artist": "An artist such as a musician, singer, painter, writer or performer.",
    "Athlete": "An athlete or sports player.",
    "OfficeHolder": "A politician or holder of a public office, such as a legislator, minister or mayor.",
    "MeanOfTransportation": "A vehicle or means of transportation, such as a ship, aircraft, car or locomotive.",
    "Building": "A building or structure, such as a house, church, tower or historic site.",
    "NaturalPlace": "A natural place, such as a mountain, river, lake, island or glacier.",
    "Village": "A village or small settlement.",
    "Animal": "An animal species, such as an insect, bird, fish, mammal or mollusc.",
    "Plant": "A plant species, such as a tree, flower, fern, moss or fungus-like plant.",
    "Album": "A music album or recording.",
    "Film": "A film or movie.",
    "WrittenWork": "A written work, such as a book, novel, magazine, newspaper or journal.",
}

IMDB = {
    "neg": "A negative movie review: the reviewer disliked the film overall.",
    "pos": "A positive movie review: the reviewer liked the film overall.",
}

EMOTION = {
    "sadness": "The author expresses sadness, grief, loneliness or feeling down.",
    "joy": "The author expresses joy, happiness, contentment or satisfaction.",
    "love": "The author expresses love, affection, tenderness or caring towards someone or something.",
    "anger": "The author expresses anger, irritation, resentment or frustration.",
    "fear": "The author expresses fear, anxiety, nervousness or worry.",
    "surprise": "The author expresses surprise, amazement or astonishment.",
}


def _banking77_descriptions(names):
    # 77 intents: descriptions are derived mechanically from the source label names,
    # so they carry no extra hand-written hints. Replace them for a tuned prompt study.
    return {n: f"A banking customer-support message whose intent is: {n.replace('_', ' ')}." for n in names}


def _clean_imdb(text):
    return re.sub(r"<br\s*/?>", "\n", text)


# repos: candidates tried in order; the one used is recorded in dataset metadata.
DATASETS = {
    "ag_news": {
        "title": "AG News (4 news topics)",
        "repos": ["fancyzhx/ag_news"], "labels": AG_NEWS,
        "text": lambda r: r["text"],
    },
    "dbpedia_14": {
        "title": "DBpedia-14 (14 Wikipedia entity types)",
        "repos": ["fancyzhx/dbpedia_14"], "labels": DBPEDIA_14,
        "text": lambda r: f"{r['title'].strip()}. {r['content'].strip()}",
    },
    "imdb": {
        "title": "IMDB movie-review sentiment (binary)",
        "repos": ["stanfordnlp/imdb"], "labels": IMDB,
        "text": lambda r: _clean_imdb(r["text"]),
        "splits": ["train", "test"],  # skip the unlabeled split
    },
    "emotion": {
        "title": "Emotion (6 emotions in English tweets)",
        "repos": ["dair-ai/emotion"], "labels": EMOTION,
        "text": lambda r: r["text"],
    },
    "banking77": {
        "title": "Banking77 (77 fine-grained customer intents)",
        "repos": ["PolyAI/banking77", "mteb/banking77"], "labels": _banking77_descriptions,
        "text": lambda r: r["text"],
    },
}


def load_hf(repo, revision, cache_dir):
    """Return ({split: [row dicts]}, label names by integer id, resolved commit)."""
    try:
        from datasets import ClassLabel, load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError("Public datasets need the optional dependency: pip install -e '.[public]'") from exc
    sha = revision or HfApi().dataset_info(repo).sha
    ds = load_dataset(repo, revision=sha, cache_dir=cache_dir)
    first = next(iter(ds.values()))
    feature = first.features.get("label")
    names = feature.names if isinstance(feature, ClassLabel) else None
    splits = {}
    for split, part in ds.items():
        rows = part.to_list()
        if names is None:
            if not rows or "label_text" not in rows[0]:
                raise ValueError(f"{repo}: cannot resolve label names (no ClassLabel or label_text)")
            for r in rows:
                r["label_name"] = r["label_text"]
        else:
            for r in rows:
                r["label_name"] = names[int(r["label"])]
        splits[split] = rows
    return splits, names, sha


def fetch(name, cache_dir, revision=None):
    """Raw rows with source_split in train/validation/test plus label descriptions and provenance."""
    spec = DATASETS[name]
    errors = []
    for repo in spec["repos"]:
        try:
            splits, names, sha = load_hf(repo, revision, cache_dir)
            break
        except ValueError:
            raise
        except Exception as exc:  # network, missing repo, legacy loading script
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
    else:
        raise ValueError(f"Could not load {name}; tried " + " | ".join(errors))
    keep = spec.get("splits") or list(splits)
    if "test" not in keep:
        raise ValueError(f"{repo} has no labeled test split")
    observed = sorted({r["label_name"] for s in keep for r in splits[s]})
    labels = spec["labels"](observed) if callable(spec["labels"]) else dict(spec["labels"])
    if set(observed) != set(labels):
        raise ValueError(f"{repo} label names changed: {observed}")
    raw = []
    # Non-test splits first, so an exact duplicate of a training text is dropped from test.
    for split in sorted(keep, key=lambda s: s == "test"):
        raw.extend({"text": spec["text"](r), "label": r["label_name"],
                    "source_split": "test" if split == "test" else "train"} for r in splits[split])
    provenance = {"hub_repo": repo, "hub_revision": sha, "source_splits": keep, "title": spec["title"]}
    return raw, labels, provenance
