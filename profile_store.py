import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np

from taste_survey import build_style_embedding_from_survey
from user_profile import (
    fetch_wine_profile_from_gpt,
    build_wine_embedding_text,
    get_embedding,
)

DATA_DIR = "data"
USERS_DIR = os.path.join(DATA_DIR, "users")
WINES_PATH = os.path.join(DATA_DIR, "wines.json")

os.makedirs(USERS_DIR, exist_ok=True)


# ====== basic file helpers ======

def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ====== wines catalog ======

def load_wines_db() -> Dict[str, Any]:
    data = _load_json(WINES_PATH)
    return data or {}


def save_wines_db(db: Dict[str, Any]) -> None:
    _save_json(WINES_PATH, db)


def normalize_wine_id(name: str) -> str:
    return (
        name.lower()
        .replace("&", "and")
        .replace("/", " ")
        .replace("'", "")
        .replace('"', "")
        .replace(".", "")
        .replace(",", "")
        .strip()
        .replace(" ", "_")
    )


def upsert_wine_profile(
    wines_db: Dict[str, Any],
    wine_name: str,
    profile: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Ensure a wine is in the wines_db with embedding + embedding_text.
    Returns (wine_id, updated_db).
    """
    wine_id = normalize_wine_id(profile.get("resolved_name") or wine_name)
    existing = wines_db.get(wine_id)

    # build embedding text + vector
    from user_profile import build_wine_embedding_text as _build_wine_embedding_text

    emb_text = _build_wine_embedding_text(profile)
    emb = get_embedding(emb_text)  # normalized

    wines_db[wine_id] = {
        "wine_id": wine_id,
        "name": profile.get("resolved_name") or profile.get("input_name") or wine_name,
        "producer": profile.get("producer"),
        "country": profile.get("country"),
        "region": profile.get("region"),
        "appellation": profile.get("appellation"),
        "color": profile.get("color"),
        "grapes": profile.get("grapes") or [],
        "embedding_text": emb_text,
        "embedding": emb.tolist(),
    }
    return wine_id, wines_db


# ====== user profile ======

def get_user_path(user_id: str) -> str:
    return os.path.join(USERS_DIR, f"{user_id}.json")


import json
from pathlib import Path
from typing import Any, Dict

PROFILE_DIR = Path("data/users")  # whatever you’re using

USER_VEC_DIM = 3072  # whatever your embedding size is

def _user_path(user_id: str) -> Path:
    return PROFILE_DIR / f"{user_id}.json"

def _default_user(user_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "survey_answers": None,
        "style_vec": [0.0] * USER_VEC_DIM,
        "favorite_wines": [],     # 👈 IMPORTANT
        "tastings": [],           # optional but useful
    }

def load_user_profile(user_id: str) -> Dict[str, Any]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _user_path(user_id)

    if not path.exists():
        user = _default_user(user_id)
        save_user_profile(user)
        return user
    with path.open("r", encoding="utf-8") as f:
        user = json.load(f)

    if "favorite_wines" not in user:
        user["favorite_wines"] = []
    if "tastings" not in user:           # 👈 ensure key exists
        user["tastings"] = []

    return user

def save_user_profile(user: Dict[str, Any]) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _user_path(user["user_id"])
    with path.open("w", encoding="utf-8") as f:
        json.dump(user, f, indent=2)



# def save_user_profile(user: Dict[str, Any]) -> None:
#     _save_json(get_user_path(user["user_id"]), user)


# ====== building & updating taste vectors ======
def recompute_user_vec(user: Dict[str, Any], wines_db: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recompute user_vec from:
      - style_vec (survey)  --> weak prior
      - favorite wines      --> base positive signal
      - tastings            --> weighted by rating, including negative weights

    We do a weighted average in embedding space, then L2-normalize.
    """

    vectors: List[np.ndarray] = []
    weights: List[float] = []

    # ---------- 1) Survey style vector (weak) ----------
    style_vec = user.get("style_vec")
    if style_vec is not None:
        v = np.array(style_vec, dtype=np.float32)
        if np.linalg.norm(v) > 0:
            vectors.append(v)
            # Survey is a prior, not dominant
            weights.append(0.3)

    # ---------- 2) Favorites (photo/text) ----------
    # These are wines the user explicitly said they liked.
    for fav in user.get("favorite_wines", []):
        wid = fav["wine_id"]
        wine = wines_db.get(wid)
        if not wine:
            continue
        emb_list = wine.get("embedding")
        if not emb_list:
            continue

        v = np.array(emb_list, dtype=np.float32)
        if np.linalg.norm(v) == 0:
            continue

        src = fav.get("source", "text")
        # Slightly boost photo-based favorites (stronger signal that they really drank it)
        if src == "photo":
            w = 1.2
        else:
            w = 1.0

        vectors.append(v)
        weights.append(w)

    # ---------- 3) Tastings with ratings ----------
    # Aggregate max rating per wine (most informative).
    rating_by_wine: Dict[str, int] = {}
    for t in user.get("tastings", []):
        wid = t.get("wine_id")
        if not wid:
            continue
        rating = int(t.get("rating", 0))
        rating_by_wine[wid] = max(rating_by_wine.get(wid, 0), rating)

    # Map 1–5 rating to weights (including negative pull-away for bad ones).
    rating_to_weight = {
        1: -0.8,  # strongly push away from 1★ wines
        2: -0.3,
        3: 0.3,
        4: 1.0,
        5: 2.0,   # big pull towards phenomenal wines
    }

    for wid, rating in rating_by_wine.items():
        wine = wines_db.get(wid)
        if not wine:
            continue
        emb_list = wine.get("embedding")
        if not emb_list:
            continue

        v = np.array(emb_list, dtype=np.float32)
        if np.linalg.norm(v) == 0:
            continue

        w = rating_to_weight.get(int(rating), 0.0)
        if w == 0.0:
            continue

        vectors.append(v)
        weights.append(w)

    # ---------- 4) If nothing, bail ----------
    if not vectors:
        # No usable info; store empty and return
        user["user_vec"] = None
        return user

    # Stack and compute weighted average
    vecs = np.stack(vectors, axis=0)
    ws = np.array(weights, dtype=np.float32).reshape(-1, 1)

    weighted = (vecs * ws).sum(axis=0) / ws.sum()

    # Normalize
    norm = np.linalg.norm(weighted)
    if norm > 0:
        weighted = weighted / norm

    user["user_vec"] = weighted.tolist()
    return user


# ====== high-level operations ======

def set_survey_for_user(user_id: str, survey_answers: Dict[str, Any]) -> Dict[str, Any]:
    user = load_user_profile(user_id)
    style_vec, _ = build_style_embedding_from_survey(survey_answers)
    user["survey_answers"] = survey_answers
    user["style_vec"] = style_vec.tolist()

    wines_db = load_wines_db()
    user = recompute_user_vec(user, wines_db)
    save_user_profile(user)
    return user


def add_favorite_wine_by_name(user_id: str, wine_name: str) -> Dict[str, Any]:
    user = load_user_profile(user_id)
    wines_db = load_wines_db()

    profile = fetch_wine_profile_from_gpt(wine_name)
    wine_id, wines_db = upsert_wine_profile(wines_db, wine_name, profile)
    save_wines_db(wines_db)

    # append if not already
    if not any(f["wine_id"] == wine_id for f in user["favorite_wines"]):
        user["favorite_wines"].append(
            {
                "wine_id": wine_id,
                "source": "text",
                "added_at": datetime.utcnow().isoformat() + "Z",
            }
        )

    user = recompute_user_vec(user, wines_db)
    save_user_profile(user)
    return user


from datetime import datetime
from typing import Dict, Any

def add_favorite_wine_from_profile(user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    user = load_user_profile(user_id)
    wines_db = load_wines_db()

    resolved_name = profile.get("resolved_name") or profile.get("name")
    if not resolved_name:
        raise ValueError("Profile missing resolved_name/name")

    wine_id = normalize_wine_id(resolved_name)

    # --- write / update the wine record in wines_db ---
    existing = wines_db.get(wine_id, {})

    wine_record = {
        **existing,
        "wine_id": wine_id,
        "name": resolved_name,
        "producer": profile.get("producer") or existing.get("producer"),
        "country": profile.get("country") or existing.get("country"),
        "region": profile.get("region") or existing.get("region"),
        "appellation": profile.get("appellation") or existing.get("appellation"),
        "color": profile.get("color") or existing.get("color"),
        "grapes": profile.get("grapes") or existing.get("grapes") or [],
        "embedding_text": profile.get("embedding_text") or existing.get("embedding_text"),
        "embedding": profile.get("embedding") or existing.get("embedding"),
        # ✅ NEW: carry the image into the wine record
        "image_base64": profile.get("image_base64") or existing.get("image_base64"),
        "image_url": profile.get("image_url") or existing.get("image_url"),
    }

    wines_db[wine_id] = wine_record
    save_wines_db(wines_db)

    # --- append to the user’s favorites list ---
    favorites = user.get("favorite_wines") or []

    favorites.append(
        {
            "wine_id": wine_id,
            "display_name": resolved_name,
            "source": profile.get("source", "photo"),
            "added_at": datetime.utcnow().isoformat(timespec="seconds"),
            # optional thumbnail for favorites list
            "image_base64": profile.get("image_base64"),
            "image_url": profile.get("image_url"),
        }
    )

    user["favorite_wines"] = favorites
    save_user_profile(user)
    return user



def add_tasting(
    user_id: str,
    wine_id: str,
    rating: int,
    context: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    After dinner: 'I tried this wine and I rate it X'.
    """
    user = load_user_profile(user_id)
    wines_db = load_wines_db()

    user.setdefault("tastings", []).append(
        {
            "wine_id": wine_id,
            "date": datetime.utcnow().date().isoformat(),
            "rating": rating,
            "context": context,
            "notes": notes,
        }
    )

    user = recompute_user_vec(user, wines_db)
    save_user_profile(user)
    return user
