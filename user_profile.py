import os
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 1. OPENAI CLIENT + EMBEDDINGS
# =========================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment")

client = OpenAI(api_key=API_KEY)


def get_embedding(text: str) -> np.ndarray:
    """
    Generate a text embedding using OpenAI's embedding model.
    Returns a normalized numpy vector.
    """
    if not isinstance(text, str) or not text.strip():
        text = ""

    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# =========================
# 2. GPT: FETCH WINE PROFILE
# =========================

def fetch_wine_profile_from_gpt(wine_name: str) -> Dict[str, Any]:
    """
    Use GPT-4.1 to fetch a structured profile for a wine.

    Returns a dict like:
    {
      "input_name": "...",
      "resolved_name": "...",
      "producer": "...",
      "country": "...",
      "region": "...",
      "appellation": "...",
      "color": "red/white/rosé/orange/sparkling/other",
      "grapes": [...],
      "vintage_typical": "e.g. 2019" or null,
      "body": "very light/light/medium/full/very full",
      "acidity": "very low/low/medium/high/very high",
      "tannin": "very low/low/medium/high/very high",
      "sweetness": "dry/off-dry/medium-sweet/sweet",
      "oak": "unoaked/light/medium/heavy",
      "style_description": "... short tasting note ...",
      "not_found": bool
    }
    """
    system_prompt = (
        "You are an expert sommelier and wine database. "
        "Given the name of a wine that a user enjoyed (often including producer and region), "
        "return a structured profile describing what the wine is like. "
        "If the exact producer/cuvée is unknown, infer the most likely style based on grape and region.\n\n"
        "You MUST respond in JSON only, matching the provided schema. "
    )

    schema = {
        "type": "object",
        "properties": {
            "input_name": {"type": "string"},
            "resolved_name": {"type": ["string", "null"]},
            "producer": {"type": ["string", "null"]},
            "country": {"type": ["string", "null"]},
            "region": {"type": ["string", "null"]},
            "appellation": {"type": ["string", "null"]},
            "color": {
                "type": ["string", "null"],
                "enum": ["red", "white", "rosé", "rose", "orange", "sparkling", "other", None],
            },
            "grapes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "vintage_typical": {"type": ["string", "null"]},
            "body": {
                "type": ["string", "null"],
                "enum": ["very light", "light", "medium", "full", "very full", None],
            },
            "acidity": {
                "type": ["string", "null"],
                "enum": ["very low", "low", "medium", "high", "very high", None],
            },
            "tannin": {
                "type": ["string", "null"],
                "enum": ["very low", "low", "medium", "high", "very high", None],
            },
            "sweetness": {
                "type": ["string", "null"],
                "enum": ["dry", "off-dry", "medium-sweet", "sweet", None],
            },
            "oak": {
                "type": ["string", "null"],
                "enum": ["unoaked", "light", "medium", "heavy", None],
            },
            "style_description": {"type": ["string", "null"]},
            "not_found": {"type": "boolean"},
        },
        "required": ["input_name", "grapes", "not_found"],
    }

    resp = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Wine name: {wine_name}",
                    }
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "wine_profile",
                "schema": schema,
                "strict": False,
            }
        },
        temperature=0.1,
    )

    # Extract JSON text from Responses API
    if not getattr(resp, "output", None):
        raise RuntimeError("No output from GPT-4.1")

    texts = []
    for out in resp.output:
        for part in out.content:
            if part.type == "output_text":
                texts.append(part.text)

    if not texts:
        raise RuntimeError("No output_text in GPT-4.1 response")

    raw = texts[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try salvage if model wrapped JSON in extra text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = raw[start : end + 1]
            data = json.loads(cleaned)
        else:
            raise

    # Inject input_name if missing
    data.setdefault("input_name", wine_name)
    return data


# =========================
# 3. BUILD EMBEDDING TEXT
# =========================

def build_wine_embedding_text(profile: Dict[str, Any]) -> str:
    """
    Turn a structured wine profile into a single descriptive text string
    to feed into the embedding model. The goal is to encode style, structure,
    grape, region, etc. in a consistent way.
    """
    name = profile.get("resolved_name") or profile.get("input_name") or ""
    producer = profile.get("producer")
    country = profile.get("country")
    region = profile.get("region")
    appellation = profile.get("appellation")
    color = profile.get("color")
    grapes = profile.get("grapes") or []
    vintage_typical = profile.get("vintage_typical")
    body = profile.get("body")
    acidity = profile.get("acidity")
    tannin = profile.get("tannin")
    sweetness = profile.get("sweetness")
    oak = profile.get("oak")
    style_description = profile.get("style_description")

    parts = []

    base = name
    if producer and producer.lower() not in name.lower():
        base = f"{producer} {name}"
    parts.append(base)

    loc_bits = []
    if appellation:
        loc_bits.append(appellation)
    elif region:
        loc_bits.append(region)
    if country:
        loc_bits.append(country)
    if loc_bits:
        parts.append(" from " + ", ".join(loc_bits))

    if color:
        parts.append(f". This is a {color} wine")

    if grapes:
        parts.append(f" made from {', '.join(grapes)}")

    if vintage_typical:
        parts.append(f", typically around vintage {vintage_typical}")

    structure_bits = []
    if body:
        structure_bits.append(f"body: {body}")
    if acidity:
        structure_bits.append(f"acidity: {acidity}")
    if tannin and color in ["red", "rosé", "rose", "orange"]:
        structure_bits.append(f"tannin: {tannin}")
    if sweetness:
        structure_bits.append(f"sweetness: {sweetness}")
    if oak:
        structure_bits.append(f"oak: {oak}")
    if structure_bits:
        parts.append(". Structure: " + ", ".join(structure_bits))

    if style_description:
        parts.append(". Style: " + style_description)

    text = "".join(parts)
    return text.strip()


# =========================
# 4. USER FLAVOR VECTOR
# =========================

def build_user_flavor_vector(
    favorite_wine_names: List[str]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Given a list of wine names the user enjoys, fetch profiles for each,
    build embedding texts, embed them, and average into a single user flavor vector.

    Returns:
        user_vector: np.ndarray (normalized)
        profiles: list of dicts, each profile with an added 'embedding_text' field
    """
    if not favorite_wine_names:
        raise ValueError("favorite_wine_names must not be empty")

    embeddings = []
    profiles: List[Dict[str, Any]] = []

    for name in favorite_wine_names:
        print(f"[INFO] Fetching profile for favorite wine: {name!r}")
        profile = fetch_wine_profile_from_gpt(name)
        if profile.get("not_found"):
            print(f"[WARN] Wine not found or ambiguous: {name!r}. Skipping.")
            continue

        emb_text = build_wine_embedding_text(profile)
        profile["embedding_text"] = emb_text

        print(f"[DEBUG] Embedding text for {name!r}: {emb_text}")

        emb = get_embedding(emb_text)
        embeddings.append(emb)
        profiles.append(profile)

    if not embeddings:
        raise RuntimeError("No valid wine profiles to build user vector from")

    user_vec = np.mean(embeddings, axis=0)
    # Normalize
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec, profiles


# =========================
# 5. SIMPLE CLI FOR TESTING
# =========================

def _interactive_collect_favorites(max_wines: int = 5) -> List[str]:
    """
    Simple CLI helper: prompt user to enter up to max_wines they love.
    """
    print(f"Enter up to {max_wines} wines you have enjoyed (blank line to stop):")
    names: List[str] = []
    for i in range(max_wines):
        name = input(f"Wine {i+1}: ").strip()
        if not name:
            break
        names.append(name)
    return names

import base64
from typing import BinaryIO

def fetch_wine_profile_from_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Use GPT-4.1 Vision to identify a wine from a bottle/label photo.
    Returns a profile similar to fetch_wine_profile_from_gpt, but with
    an extra 'confidence' flag and possibly less precise info.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    system_prompt = (
        "You are an expert at recognizing wine labels from photos. "
        "Given a photo of a wine bottle or label, identify the wine and return a structured JSON profile. "
        "If the exact cuvée or vintage is unclear, make your best guess but mark uncertain fields."
    )

    schema = {
        "type": "object",
        "properties": {
            "resolved_name": {"type": ["string", "null"]},
            "producer": {"type": ["string", "null"]},
            "country": {"type": ["string", "null"]},
            "region": {"type": ["string", "null"]},
            "appellation": {"type": ["string", "null"]},
            "color": {
                "type": ["string", "null"],
                "enum": ["red", "white", "rosé", "rose", "orange", "sparkling", "other", None],
            },
            "grapes": {"type": "array", "items": {"type": "string"}},
            "vintage_typical": {"type": ["string", "null"]},
            "confidence": {"type": "string"},  # e.g. 'high', 'medium', 'low'
            "notes": {"type": ["string", "null"]},
        },
        "required": ["resolved_name", "confidence"],
    }

    resp = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url,
                    }
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "wine_from_image",
                "schema": schema,
                "strict": False,
            }
        },
        temperature=0.1,
    )

    # Extract JSON as in other helpers...
    if not getattr(resp, "output", None):
        raise RuntimeError("No output from GPT-4.1")

    texts = []
    for out in resp.output:
        for part in out.content:
            if part.type == "output_text":
                texts.append(part.text)

    if not texts:
        raise RuntimeError("No output_text in GPT-4.1 response from image")

    raw = texts[0].strip()
    # same JSON parsing salvage pattern as before:
    import json
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = raw[start: end+1]
            data = json.loads(cleaned)
        else:
            raise

    return data

if __name__ == "__main__":
    # For quick manual testing:
    favorites = _interactive_collect_favorites(max_wines=5)
    if not favorites:
        print("No wines entered. Exiting.")
        raise SystemExit(0)

    user_vec, profiles = build_user_flavor_vector(favorites)

    print("\n================ USER FLAVOR VECTOR (first 10 dims) ================")
    print(user_vec[:10])
    print(f"Vector length: {len(user_vec)}\n")

    print("================ INDIVIDUAL WINE PROFILES ================\n")
    for p in profiles:
        print(f"Input name:    {p.get('input_name')}")
        print(f"Resolved name: {p.get('resolved_name')}")
        print(f"Producer:      {p.get('producer')}")
        print(f"Country:       {p.get('country')}")
        print(f"Region/App:    {p.get('region')} / {p.get('appellation')}")
        print(f"Color:         {p.get('color')}")
        print(f"Grapes:        {', '.join(p.get('grapes') or [])}")
        print(f"Body:          {p.get('body')}")
        print(f"Acidity:       {p.get('acidity')}")
        print(f"Tannin:        {p.get('tannin')}")
        print(f"Sweetness:     {p.get('sweetness')}")
        print(f"Oak:           {p.get('oak')}")
        print(f"Description:   {p.get('style_description')}")
        print(f"Embedding txt: {p.get('embedding_text')}")
        print("-" * 60)
