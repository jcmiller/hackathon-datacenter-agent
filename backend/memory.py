import json, math, os
from typing import Optional

_SOP_VECTORS = "data/sop_vectors.json"


def _embed(text: str) -> Optional[list[float]]:
    """Embed text via Gemini embedding model. Returns None if API unavailable."""
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        result = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return list(result.embeddings[0].values)
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_vectors() -> list[Optional[list[float]]]:
    if not os.path.exists(_SOP_VECTORS):
        return []
    with open(_SOP_VECTORS) as f:
        return json.load(f)


def _save_vectors(vectors: list[Optional[list[float]]]) -> None:
    os.makedirs(os.path.dirname(_SOP_VECTORS) or ".", exist_ok=True)
    with open(_SOP_VECTORS, "w") as f:
        json.dump(vectors, f)


def _record_text(r: dict) -> str:
    return f"{r.get('type','')} {r.get('summary','')} {r.get('resolution','')}".strip()


def search_incidents(query: str, path: str = "data/sop.json", top_k: int = 3) -> list[dict]:
    """Semantic search over past incidents. Falls back to substring match if embeddings unavailable."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        records = json.load(f)
    if not records:
        return []

    vectors = _load_vectors()

    # Fill in any missing vectors (new entries since last run)
    changed = False
    while len(vectors) < len(records):
        vec = _embed(_record_text(records[len(vectors)]))
        vectors.append(vec)
        changed = True
    if changed:
        _save_vectors(vectors)

    query_vec = _embed(query)
    if query_vec is None:
        # Fallback: substring keyword match
        q = query.lower()
        hits = [r for r in records if q in _record_text(r).lower()]
        return hits[:top_k]

    scored = [
        (i, _cosine(query_vec, vectors[i]))
        for i in range(len(records))
        if vectors[i] is not None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        {"similarity": round(s, 3), **records[i]}
        for i, s in scored[:top_k]
        if s > 0.4
    ]


def append_incident(record: dict, path: str = "data/sop.json") -> None:
    records = []
    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
    records.append(record)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

    # Embed and store the new entry's vector
    vectors = _load_vectors()
    vec = _embed(_record_text(record))
    vectors.append(vec)
    _save_vectors(vectors)
