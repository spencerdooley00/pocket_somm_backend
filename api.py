# api.py

import base64
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from taste_survey import build_style_embedding_from_survey
from user_profile import fetch_wine_profile_from_image, get_embedding
from menu_recommender import score_menu_for_user, build_menu_wine_embedding_text
from wine_recommender import call_gpt41_extract_wines
from profile_store import (
    load_user_profile,
    save_user_profile,
    load_wines_db,
    save_wines_db,
    set_survey_for_user,
    add_favorite_wine_by_name,
    add_favorite_wine_from_profile,
    add_tasting,
    normalize_wine_id,
)

# =========================
# Pydantic models
# =========================

class SurveyAnswers(BaseModel):
    favorite_styles: List[str]
    tannin_pref: str
    acidity_pref: str
    oak_pref: str
    adventure_pref: str


class AddFavoriteByNameBody(BaseModel):
    wine_name: str


class PhotoBody(BaseModel):
    image_base64: str
    content_type: Optional[str] = "image/jpeg"


class MenuPdfBody(BaseModel):
    pdf_base64: str


class MenuTextBody(BaseModel):
    menu_text: str


class TastingBody(BaseModel):
    wine_id: str
    rating: float
    context: Optional[str] = None
    notes: Optional[str] = None


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Wine Recommender API", version="0.1.0")

# For local dev + simulator / device HTTP calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helpers
# =========================

def _require_user_vec(user: Dict[str, Any]) -> np.ndarray:
    user_vec_list = user.get("user_vec")
    if not user_vec_list:
        raise HTTPException(
            status_code=400,
            detail="User flavor vector is empty. Complete the survey and/or add some favorite wines first.",
        )
    user_vec = np.array(user_vec_list, dtype=np.float32)
    if np.linalg.norm(user_vec) == 0:
        raise HTTPException(
            status_code=400,
            detail="User flavor vector has zero norm. Something went wrong in profile computation.",
        )
    return user_vec

# =========================
# Endpoints
# =========================

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/user/{user_id}")
def get_user(user_id: str) -> Dict[str, Any]:
    """
    Fetch the raw user profile JSON.
    """
    user = load_user_profile(user_id)
    return user


@app.post("/user/{user_id}/survey")
def update_survey(user_id: str, answers: SurveyAnswers) -> Dict[str, Any]:
    """
    Store survey answers and recompute user_vec.
    """
    survey_dict = answers.model_dump()
    user = set_survey_for_user(user_id, survey_dict)
    return {"status": "ok", "user": user}


@app.post("/user/{user_id}/favorite/by-name")
def add_favorite_text(user_id: str, body: AddFavoriteByNameBody) -> Dict[str, Any]:
    """
    Add a favorite wine by text name (user types a bottle they liked).
    """
    user = add_favorite_wine_by_name(user_id, body.wine_name)
    return {"status": "ok", "user": user}


@app.post("/user/{user_id}/favorite/from-photo")
def add_favorite_photo(user_id: str, body: PhotoBody) -> Dict[str, Any]:
    """
    Add a favorite wine from a bottle/label photo.
    iOS: pick/take photo → base64 → POST here.
    """
    try:
        image_bytes = base64.b64decode(body.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # 1) Use GPT-4.1 Vision to identify the wine and build a profile
    profile = fetch_wine_profile_from_image(image_bytes)

    # ✅ NEW: attach the raw image to the profile we pass downstream
    profile["image_base64"] = body.image_base64
    profile["source"] = "photo"

    # 2) Persist as favorite and recompute user_vec
    user = add_favorite_wine_from_profile(user_id, profile)

    return {
        "status": "ok",
        "wine_profile": profile,
        "user": user,
    }



@app.post("/user/{user_id}/tasting")
def add_tasting_event(user_id: str, body: TastingBody) -> Dict[str, Any]:
    """
    After-dinner flow: user rates a specific wine they had.
    wine_id should be one of the IDs stored in wines_db.
    """
    user = add_tasting(
        user_id=user_id,
        wine_id=body.wine_id,
        rating=body.rating,
        context=body.context,
        notes=body.notes,
    )
    return {"status": "ok", "user": user}


@app.post("/user/{user_id}/menu/pdf")
def recommend_from_menu_pdf(user_id: str, body: MenuPdfBody) -> Dict[str, Any]:
    """
    Main restaurant flow: upload a menu PDF, get ranked recommendations.
    Client sends the PDF as base64-encoded bytes.
    """
    user = load_user_profile(user_id)
    user_vec = _require_user_vec(user)

    # Decode and write PDF to a temp file so we can reuse call_gpt41_extract_wines
    try:
        pdf_bytes = base64.b64decode(body.pdf_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF")

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # Extract wines from menu
        menu_wines = call_gpt41_extract_wines(tmp_path)
        if not menu_wines:
            raise HTTPException(status_code=400, detail="No wines extracted from menu")

        # Score wines
        results = score_menu_for_user(user_vec, menu_wines)

        # Upsert wines into wines_db so they can be referenced later (tastings, etc.)
        wines_db = load_wines_db()
        session_menu_wines: List[Dict[str, str]] = []

        for item in results:
            w = item["wine"]
            name = w.get("name") or "Unknown wine"
            vintage = w.get("vintage") or ""
            region = w.get("region") or ""

            raw_id = f"{name} {vintage} {region}"
            wine_id = normalize_wine_id(raw_id)

            emb_text = build_menu_wine_embedding_text(w)
            emb = get_embedding(emb_text)

            wines_db[wine_id] = {
                "wine_id": wine_id,
                "name": name,
                "producer": w.get("producer"),
                "country": w.get("country"),
                "region": w.get("region"),
                "appellation": None,
                "color": w.get("color"),
                "grapes": w.get("grapes") or [],
                "embedding_text": emb_text,
                "embedding": emb.tolist(),
                # placeholder for now; later you can actually fetch a real label image
                "image_url": w.get("image_url") or default_image_for_color(w.get("color")),

            }


            # Label to show client
            label_parts = [name]
            if vintage:
                label_parts.append(str(vintage))
            if region:
                label_parts.append(f"({region})")
            if w.get("price_string"):
                label_parts.append(f"({w['price_string']})")
            label = " ".join(label_parts)

            session_menu_wines.append(
                {
                    "wine_id": wine_id,
                    "label": label,
                }
            )

        save_wines_db(wines_db)

        # Strip embeddings from results before returning to keep payload lighter
        cleaned_results = [
            {
                "wine": r["wine"],
                "score": r["score"],
                "embedding_text": r.get("embedding_text"),
            }
            for r in results
        ]

        return {
            "status": "ok",
            "results": cleaned_results,
            "menu_wines": session_menu_wines,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/wine/{wine_id}")
def get_wine_detail(wine_id: str) -> Dict[str, Any]:
    wines_db = load_wines_db()
    wine = wines_db.get(wine_id)
    if wine is None:
        raise HTTPException(status_code=404, detail=f"Wine not found: {wine_id}")
    wine_copy = dict(wine)
    wine_copy.pop("embedding", None)
    return wine_copy



@app.get("/wine/{wine_id}/similar")
def similar_wines(wine_id: str, top_k: int = 5):
    wines = load_wines_db()
    if wine_id not in wines:
        raise HTTPException(status_code=404, detail="Wine not found")

    target = np.array(wines[wine_id]["embedding"], dtype=np.float32)
    target_norm = target / (np.linalg.norm(target) + 1e-10)

    scored = []
    for w_id, w in wines.items():
        if w_id == wine_id:
            continue
        emb = np.array(w["embedding"], dtype=np.float32)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        score = float(np.dot(target_norm, emb_norm))
        scored.append((score, w))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return {
        "wine_id": wine_id,
        "similar": [
            {
                "wine_id": item["wine_id"],
                "name": item["name"],
                "producer": item.get("producer"),
                "region": item.get("region"),
                "score": score
            }
            for score, item in top
        ]
    }

from collections import Counter
from typing import Tuple
# ... existing imports ...

def _collect_wine_stats_for_user(user: Dict[str, Any]) -> Dict[str, Any]:
    wines_db = load_wines_db()
    favorites = user.get("favorite_wines", []) or []
    tastings = user.get("tastings", []) or []

    grape_counter = Counter()
    country_counter = Counter()
    region_counter = Counter()
    color_counter = Counter()

    # Count favorites
    for fav in favorites:
        wine_id = fav.get("wine_id")
        wine = wines_db.get(wine_id)
        if not wine:
            continue
        for g in wine.get("grapes") or []:
            grape_counter[g] += 1
        if wine.get("country"):
            country_counter[wine["country"]] += 1
        if wine.get("region"):
            region_counter[wine["region"]] += 1
        if wine.get("color"):
            color_counter[wine["color"]] += 1

    # You could also weight by tastings/rating later if you want
    stats = {
        "total_favorites": len(favorites),
        "total_tastings": len(tastings),
        "top_grapes": [{"name": g, "count": c} for g, c in grape_counter.most_common(5)],
        "top_countries": [{"name": c, "count": n} for c, n in country_counter.most_common(5)],
        "top_regions": [{"name": r, "count": n} for r, n in region_counter.most_common(5)],
        "top_colors": [{"name": col, "count": n} for col, n in color_counter.most_common(3)],
    }
    return stats


def _build_text_summary(user: Dict[str, Any], stats: Dict[str, Any]) -> str:
    survey = user.get("survey_answers") or {}

    styles = survey.get("favorite_styles") or []
    tannin_pref = survey.get("tannin_pref")
    acidity_pref = survey.get("acidity_pref")
    oak_pref = survey.get("oak_pref")

    # Very simple heuristic mapping
    style_phrases = []
    if any("light" in s for s in styles):
        style_phrases.append("light to medium-bodied reds")
    if any("medium_red" in s for s in styles):
        style_phrases.append("medium-bodied reds")
    if any("crisp_white" in s for s in styles):
        style_phrases.append("crisp, refreshing whites")

    tannin_phrase = None
    if tannin_pref == "low":
        tannin_phrase = "low tannin"
    elif tannin_pref == "medium":
        tannin_phrase = "medium tannin"
    elif tannin_pref == "high":
        tannin_phrase = "firm tannin"

    acidity_phrase = None
    if acidity_pref == "low":
        acidity_phrase = "softer acidity"
    elif acidity_pref == "medium":
        acidity_phrase = "balanced acidity"
    elif acidity_pref == "high":
        acidity_phrase = "bright, high acidity"

    oak_phrase = None
    if oak_pref == "low":
        oak_phrase = "minimal oak influence"
    elif oak_pref == "medium":
        oak_phrase = "some oak complexity"
    elif oak_pref == "high":
        oak_phrase = "noticeable oak and toast"

    parts = []

    if style_phrases:
        parts.append(f"You gravitate toward {', and '.join(style_phrases)}.")
    else:
        parts.append("Your wine preferences are still taking shape.")

    profile_bits = [p for p in [tannin_phrase, acidity_phrase, oak_phrase] if p]
    if profile_bits:
        parts.append("You tend to prefer wines with " + ", ".join(profile_bits) + ".")

    top_grapes = stats.get("top_grapes") or []
    if top_grapes:
        grape_names = [g["name"] for g in top_grapes[:3]]
        parts.append("Your favorite grapes so far include " + ", ".join(grape_names) + ".")

    top_countries = stats.get("top_countries") or []
    if top_countries:
        country_names = [c["name"] for c in top_countries[:3]]
        parts.append("You’re especially drawn to wines from " + ", ".join(country_names) + ".")

    return " ".join(parts)


@app.get("/user/{user_id}/summary")
def get_user_summary(user_id: str) -> Dict[str, Any]:
    """
    High-level taste summary + basic stats for a user.
    """
    user = load_user_profile(user_id)
    stats = _collect_wine_stats_for_user(user)
    summary_text = _build_text_summary(user, stats)

    return {
        "user_id": user_id,
        "summary_text": summary_text,
        "stats": stats,
    }

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from collections import Counter

# ...your other imports and models...
from typing import List, Optional
from pydantic import BaseModel

class WineSearchResult(BaseModel):
    wine_id: str
    name: str
    producer: Optional[str] = None
class ResolveWineNameBody(BaseModel):
    wine_name: str


class WineProfileBody(BaseModel):
    profile: Dict[str, Any]
from collections import Counter

class UserInsights(BaseModel):
    summary: str
    top_grapes: List[str]
    top_countries: List[str]
    top_regions: List[str]
    top_vintages: List[int]

@app.get("/wine_search", response_model=List[WineSearchResult])
def search_wines(q: str) -> List[WineSearchResult]:
    q_norm = q.strip().lower()
    if not q_norm:
        return []

    wines_db = load_wines_db()
    matches: List[WineSearchResult] = []

    for wine_id, wine in wines_db.items():
        name = (wine.get("name") or "").strip()
        producer = (wine.get("producer") or "").strip()
        haystack = f"{name} {producer}".lower()
        if q_norm in haystack:
            matches.append(
                WineSearchResult(
                    wine_id=wine_id,
                    name=name or wine_id,
                    producer=producer or None,
                )
            )

    return matches[:25]
from user_profile import fetch_wine_profile_from_gpt
from profile_store import add_favorite_wine_from_profile

@app.post("/wine/resolve-text")
def resolve_wine_text(body: ResolveWineNameBody) -> Dict[str, Any]:
    """
    Use GPT to resolve a free-text wine name into a structured profile,
    but DO NOT save anything yet. This is for the confirmation step.
    """
    profile = fetch_wine_profile_from_gpt(body.wine_name)

    # Optional: if GPT indicates not_found, you could return 404 later
    # if profile.get("not_found"):
    #     raise HTTPException(status_code=404, detail="Wine not found")

    return {"status": "ok", "profile": profile}
@app.post("/user/{user_id}/favorite/from-profile")
def add_favorite_from_profile_endpoint(
    user_id: str, body: WineProfileBody
) -> Dict[str, Any]:
    """
    After the user confirms a resolved wine profile, persist it as a favorite
    and recompute their taste vector.
    """
    user = add_favorite_wine_from_profile(user_id, body.profile)
    return {"status": "ok", "user": user}

def default_image_for_color(color: Optional[str]) -> Optional[str]:
    if not color:
        return None
    c = color.lower()
    if c == "red":
        return "https://example.com/red-bottle-placeholder.png"
    if c == "white":
        return "https://example.com/white-bottle-placeholder.png"
    # etc.
    return None
def _compute_user_insights(user_id: str) -> UserInsights:
    user = load_user_profile(user_id)
    wines_db = load_wines_db()

    favorites = user.get("favorite_wines") or user.get("favorites") or []
    survey = user.get("survey_answers") or {}

    grape_counts = Counter()
    country_counts = Counter()
    region_counts = Counter()
    vintage_counts = Counter()

    for fav in favorites:
        wine_id = fav.get("wine_id")
        if not wine_id:
            continue
        wine = wines_db.get(wine_id)
        if not wine:
            continue

        grapes = wine.get("grapes") or []
        for g in grapes:
            grape_counts[g.strip()] += 1

        country = (wine.get("country") or "").strip()
        if country:
            country_counts[country] += 1

        region = (wine.get("region") or "").strip()
        if region:
            region_counts[region] += 1

        # crude vintage parse from name
        name = wine.get("name") or ""
        for token in name.split():
            if token.isdigit() and len(token) == 4:
                try:
                    year = int(token)
                    if 1970 <= year <= 2050:
                        vintage_counts[year] += 1
                except ValueError:
                    pass

    def top_keys(counter: Counter, n: int = 5):
        return [k for k, _ in counter.most_common(n)]

    top_grapes = top_keys(grape_counts)
    top_countries = top_keys(country_counts)
    top_regions = top_keys(region_counts)
    top_vintages = top_keys(vintage_counts)

    # Build a simple textual summary
    tannin = survey.get("tannin_pref", "").lower() or "unknown"
    acidity = survey.get("acidity_pref", "").lower() or "unknown"
    oak = survey.get("oak_pref", "").lower() or "unknown"
    adventure = survey.get("adventure_pref", "").lower() or "unknown"
    styles = survey.get("favorite_styles") or []

    style_bits = []
    if styles:
        style_bits.append(", ".join(styles).replace("_", " "))

    parts = []

    if styles:
        parts.append(f"You tend to enjoy: {', '.join(s.replace('_', ' ') for s in styles)}.")
    parts.append(f"Your palate leans toward {tannin} tannin, {acidity} acidity, and {oak} oak.")
    parts.append(f"You're generally {adventure} on trying new styles.")

    if top_grapes:
        parts.append("You keep coming back to grapes like " + ", ".join(top_grapes) + ".")
    if top_countries:
        parts.append("Most of your favorites are from " + ", ".join(top_countries) + ".")
    if top_regions:
        parts.append("Regions that show up a lot for you: " + ", ".join(top_regions) + ".")
    if top_vintages:
        min_v = min(top_vintages)
        max_v = max(top_vintages)
        if min_v == max_v:
            parts.append(f"Your wines skew toward the {max_v} vintage.")
        else:
            parts.append(f"Your wines skew toward vintages {min_v}–{max_v}.")

    summary = " ".join(parts)

    return UserInsights(
        summary=summary,
        top_grapes=top_grapes,
        top_countries=top_countries,
        top_regions=top_regions,
        top_vintages=top_vintages,
    )
@app.get("/user/{user_id}/insights", response_model=UserInsights)
def get_user_insights(user_id: str) -> UserInsights:
    return _compute_user_insights(user_id)
