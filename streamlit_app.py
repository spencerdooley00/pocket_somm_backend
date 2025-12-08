# streamlit_app.py

import os
import tempfile
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st

# Your own modules
from taste_survey import build_style_embedding_from_survey  # used indirectly via profile_store
from wine_recommender import call_gpt41_extract_wines
from menu_recommender import build_menu_wine_embedding_text
from user_profile import get_embedding, fetch_wine_profile_from_image

from profile_store import (
    load_user_profile,
    set_survey_for_user,
    add_favorite_wine_from_profile,
    add_favorite_wine_by_name,
    load_wines_db,
    add_tasting,
    normalize_wine_id,   # <-- add
    save_wines_db,       # <-- add
)

# =========================
# HELPERS
# =========================
def score_menu_for_user_vec(
    user_vec: np.ndarray,
    menu_wines: List[Dict[str, Any]],
    favorite_embs: Optional[List[np.ndarray]] = None,
    alpha: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Hybrid scoring:
      score = alpha * cos(user_vec, wine) + (1 - alpha) * max_i cos(wine, favorite_i)

    Returns list of:
      {
        "wine": <menu wine dict>,
        "score": float,
        "embedding_text": str,
        "embedding": np.ndarray,
      }
    """
    results = []
    favorite_embs = favorite_embs or []

    for w in menu_wines:
        emb_text = build_menu_wine_embedding_text(w)
        emb = get_embedding(emb_text)  # already normalized
        base = float(np.dot(user_vec, emb))

        bonus = 0.0
        if favorite_embs:
            sims = [float(np.dot(emb, f)) for f in favorite_embs]
            bonus = max(sims)

        score = alpha * base + (1.0 - alpha) * bonus

        results.append(
            {
                "wine": w,
                "score": score,
                "embedding_text": emb_text,
                "embedding": emb,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results




def wine_matches_color_filter(wine: Dict[str, Any], choice: str) -> bool:
    if choice == "Any":
        return True

    choice_lower = choice.lower()
    color = (wine.get("color") or "").lower()
    section = (wine.get("section") or "").lower()

    if not color:
        if "red" in section:
            color = "red"
        elif "white" in section:
            color = "white"
        elif "rosé" in section or "rose" in section:
            color = "rosé"
        elif "orange" in section:
            color = "orange"
        elif "sparkling" in section or "fizzy" in section:
            color = "sparkling"

    if choice_lower == "rosé":
        return color in ["rosé", "rose"]
    return color == choice_lower


# =========================
# STREAMLIT APP
# =========================

def main():
    st.set_page_config(page_title="Wine Menu Recommender", page_icon="🍷", layout="wide")

    # --- User ID / Profile ---
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""

    st.title("🍷 Wine Menu Recommender (MVP)")

    st.markdown(
        "Create a taste profile once (survey + bottles you liked), "
        "reuse it at restaurants to get menu recommendations, "
        "and then rate what you drank to refine your profile over time."
    )

    st.markdown("### Who are you?")
    user_id = st.text_input("Profile name / ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id.strip()

    if not st.session_state.user_id:
        st.warning("Enter a profile name/ID to start (e.g. `spencer`).")
        return

    user_id = st.session_state.user_id
    user = load_user_profile(user_id)
    wines_db = load_wines_db()

    st.caption(f"Using profile file: `data/users/{user_id}.json`")

    # --- Layout ---
    col1, col2 = st.columns(2)

    # ------------------------------------------------
    # LEFT: Taste Survey (stored persistently)
    # ------------------------------------------------
    with col1:
        st.header("1️⃣ Taste survey")

        # Pre-fill from existing survey if present
        existing_survey = user.get("survey_answers") or {}

        style_options = {
            "Light, fruity reds (Beaujolais, Pinot Noir)": "light_fruity_red",
            "Medium reds (Chianti, Rioja, Cab Franc)": "medium_red",
            "Bold, rich reds (Napa Cab, Malbec, Syrah)": "bold_red",
            "Crisp, refreshing whites (Sauv Blanc, Assyrtiko)": "crisp_white",
            "Richer whites (Chardonnay, Viognier)": "rich_white",
            "Rosé": "rose",
            "Sparkling (Champagne, Prosecco, Cava)": "sparkling",
            "Orange / skin-contact wines": "orange",
            "Sweet wines (Moscato, Sauternes, sweet Riesling)": "sweet",
        }

        existing_styles = existing_survey.get("favorite_styles") or []
        default_labels = [
            label for label, key in style_options.items()
            if key in existing_styles
        ]

        st.markdown("**What kinds of wines do you usually enjoy?**")
        selected_styles_labels = st.multiselect(
            "Select all that apply:",
            list(style_options.keys()),
            default=default_labels,
        )
        favorite_styles = [style_options[label] for label in selected_styles_labels]

        tannin_default = (existing_survey.get("tannin_pref") or "medium")
        acidity_default = (existing_survey.get("acidity_pref") or "high")
        oak_default = (existing_survey.get("oak_pref") or "low")
        adventure_default = (existing_survey.get("adventure_pref") or "medium")

        st.markdown("**How do you feel about tannins?**  *(that drying, grippy feeling)*")
        tannin_pref = st.radio(
            "Tannin preference",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(tannin_default),
            format_func=lambda x: {
                "low": "I prefer very smooth wines",
                "medium": "A little grip is nice",
                "high": "I like firm, structured wines",
            }[x],
        )

        st.markdown("**How do you feel about acidity?**  *(how bright / zippy a wine feels)*")
        acidity_pref = st.radio(
            "Acidity preference",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(acidity_default),
            format_func=lambda x: {
                "low": "Soft and round",
                "medium": "Balanced freshness",
                "high": "Zippy and bright",
            }[x],
        )

        st.markdown("**How much oak flavor do you enjoy?**")
        oak_pref = st.radio(
            "Oak preference",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(oak_default),
            format_func=lambda x: {
                "low": "Little to no oak",
                "medium": "Some subtle oak",
                "high": "Clearly oaky (vanilla, toast, spice)",
            }[x],
        )

        st.markdown("**How adventurous are you?**")
        adventure_pref = st.radio(
            "Adventure preference",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(adventure_default),
            format_func=lambda x: {
                "low": "I like safe, familiar styles",
                "medium": "I'm open to some new things",
                "high": "Bring me weird and unusual wines",
            }[x],
        )

        survey_answers = {
            "favorite_styles": favorite_styles,
            "tannin_pref": tannin_pref,
            "acidity_pref": acidity_pref,
            "oak_pref": oak_pref,
            "adventure_pref": adventure_pref,
        }

        if st.button("💾 Save / update my taste survey"):
            try:
                user = set_survey_for_user(user_id, survey_answers)
                st.success("Saved your survey and updated your taste profile.")
            except Exception as e:
                st.error(f"Error saving survey: {e}")

        with st.expander("Current stored survey + vector", expanded=False):
            st.write("Survey answers:", user.get("survey_answers"))
            if user.get("style_vec") is not None:
                st.write("Style vector dim:", len(user["style_vec"]))

    # ------------------------------------------------
    # RIGHT: Favorite wines (typed + photo) + Menu upload
    # ------------------------------------------------
    with col2:
        st.header("2️⃣ Wines you've liked")

        # --- Typed wines: add to persistent profile on demand ---
        st.markdown("Add up to 3 wines you've really enjoyed (producer + wine name).")
        fav1 = st.text_input("Wine 1", "")
        fav2 = st.text_input("Wine 2", "")
        fav3 = st.text_input("Wine 3", "")

        typed_favorites = [w.strip() for w in [fav1, fav2, fav3] if w.strip()]

        if st.button("💾 Save these typed wines to my profile"):
            if not typed_favorites:
                st.warning("No typed wines to save.")
            else:
                try:
                    for name in typed_favorites:
                        user = add_favorite_wine_by_name(user_id, name)
                    st.success("Saved typed wines and updated your taste profile.")
                except Exception as e:
                    st.error(f"Error saving typed wines: {e}")

        # --- Bottle photos → persistent favorites ---
        st.markdown("---")
        st.markdown("**Or upload photos of bottles you've enjoyed:**")

        bottle_photos = st.file_uploader(
            "Upload bottle photos (labels visible if possible)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if bottle_photos and st.button("📸 Identify wines from these photos and save to my profile"):
            with st.spinner("Analyzing bottle photos..."):
                recognized = []
                for img in bottle_photos:
                    try:
                        profile = fetch_wine_profile_from_image(img.read())
                        user = add_favorite_wine_from_profile(user_id, profile)
                        recognized.append(profile)
                    except Exception as e:
                        st.error(f"Error identifying wine from image {img.name}: {e}")
                if recognized:
                    st.success(f"Added {len(recognized)} wines from photos to your favorites.")
                    st.session_state["last_photo_profiles"] = recognized

        if "last_photo_profiles" in st.session_state:
            photo_profiles = st.session_state["last_photo_profiles"]
            if photo_profiles:
                st.markdown("**Most recently recognized wines from photos:**")
                for p in photo_profiles:
                    name = p.get("resolved_name") or p.get("input_name")
                    conf = p.get("confidence")
                    st.markdown(f"- **{name}**  (confidence: {conf})")

        # --- Show current favorites from profile ---
        if user.get("favorite_wines"):
            st.markdown("---")
            st.markdown("**Wines saved in your profile:**")
            for fw in user["favorite_wines"]:
                wid = fw["wine_id"]
                w_profile = wines_db.get(wid)
                if not w_profile:
                    continue
                st.markdown(f"- **{w_profile.get('name')}**  ({w_profile.get('region')}, {w_profile.get('country')})")

        # --- Menu upload + filter + recommend button ---
        st.header("3️⃣ Upload a wine list")
        menu_file = st.file_uploader(
            "Upload a restaurant wine list PDF",
            type=["pdf"],
        )

        st.markdown("**Do you know what color of wine you want tonight?**")
        color_choice = st.selectbox(
            "Wine color filter",
            options=[
                "Any",
                "Red",
                "White",
                "Rosé",
                "Orange",
                "Sparkling",
            ],
            index=0,
        )

        st.caption("Once your profile is set and you've uploaded a menu, click below.")

        run_button = st.button("🍷 Recommend wines from this menu", type="primary")

    # ------------------------------------------------
    # 3️⃣ ACTION: Recommend wines using stored user_vec
    # ------------------------------------------------
    if run_button:
        if menu_file is None:
            st.error("Please upload a wine list PDF before running recommendations.")
            return

        # Reload user + wines in case they were just updated
        user = load_user_profile(user_id)
        wines_db = load_wines_db()

        user_vec_list = user.get("user_vec")
        if not user_vec_list:
            st.error(
                "Your taste profile (user_vec) is empty. "
                "Complete the survey and/or add some favorite wines first."
            )
            return

        user_vec = np.array(user_vec_list, dtype=np.float32)
        if np.linalg.norm(user_vec) == 0:
            st.error("Your taste vector is zero. Something went wrong in profile computation.")
            return
        favorite_embs: List[np.ndarray] = []
        for fw in user.get("favorite_wines", []):
            wid = fw["wine_id"]
            w_profile = wines_db.get(wid)
            if not w_profile:
                continue
            emb_list = w_profile.get("embedding")
            if not emb_list:
                continue
            favorite_embs.append(np.array(emb_list, dtype=np.float32))
        # Save uploaded menu to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(menu_file.read())
            tmp_path = tmp.name

        # Extract wines from menu
        try:
            menu_wines = call_gpt41_extract_wines(tmp_path)
        except Exception as e:
            st.error(f"Error extracting wines from menu via GPT: {e}")
            return
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if not menu_wines:
            st.warning("No wines were extracted from the menu. Check the PDF or the extraction logic.")
            return

        # Score menu
        try:
            results = score_menu_for_user_vec(user_vec, menu_wines, favorite_embs=favorite_embs, alpha=0.7)
        except Exception as e:
            st.error(f"Error scoring wines on the menu: {e}")
            return
        # Persist menu wines into wines_db and remember them for rating UI
        wines_db = load_wines_db()
        session_menu_wines = []  # list of (wine_id, label) for this menu

        for item in results:
            w = item["wine"]
            emb = item["embedding"]
            emb_text = item["embedding_text"]

            name = w.get("name") or "Unknown wine"
            vintage = w.get("vintage") or ""
            region = w.get("region") or ""
            # Build a reasonably unique ID
            raw_id = f"{name} {vintage} {region}"
            wine_id = normalize_wine_id(raw_id)

            wines_db[wine_id] = {
                "wine_id": wine_id,
                "name": name,
                "producer": None,  # we don't know from the menu alone
                "country": w.get("country"),
                "region": w.get("region"),
                "appellation": None,
                "color": w.get("color"),
                "grapes": w.get("grapes") or [],
                "embedding_text": emb_text,
                "embedding": emb.tolist(),
            }

            # Nice label for rating UI later
            label_parts = [name]
            if vintage:
                label_parts.append(str(vintage))
            if w.get("section"):
                label_parts.append(f"[{w['section']}]")
            if w.get("price_string"):
                label_parts.append(f"({w['price_string']})")
            label = " ".join(label_parts)

            session_menu_wines.append((wine_id, label))

        save_wines_db(wines_db)
        # Store these in session so the "After dinner" section can show them
        st.session_state["last_menu_wines"] = session_menu_wines

        # Apply color filter
        filtered_results = [r for r in results if wine_matches_color_filter(r["wine"], color_choice)]
        if not filtered_results:
            st.warning(f"No wines matched your color filter ({color_choice}). Showing all wines instead.")
            filtered_results = results

        # Display results
        st.success(f"Found {len(menu_wines)} wines on this menu and ranked them by your stored taste profile.")

        top_k = min(10, len(filtered_results))
        st.subheader(f"Top {top_k} recommendations for you:")

        display_rows = []
        for item in filtered_results[:top_k]:
            w = item["wine"]
            display_rows.append(
                {
                    "Name": w.get("name"),
                    "Vintage": w.get("vintage"),
                    "Section": w.get("section"),
                    "Color": w.get("color"),
                    "Grapes": ", ".join(w.get("grapes") or []),
                    "Region": w.get("region"),
                    "Country": w.get("country"),
                    "Price": w.get("price_string"),
                    "Similarity": round(item["score"], 3),
                }
            )

        st.dataframe(display_rows, use_container_width=True)

        with st.expander("Debug: user profile vector info"):
            st.write("user_vec_dim:", len(user_vec_list))
            st.write("favorite_wines:", user.get("favorite_wines"))
            st.write("tastings:", user.get("tastings"))

    # ------------------------------------------------
    # 4️⃣ AFTER DINNER: Rate a wine you tried
    # ------------------------------------------------
    st.markdown("---")
    st.header("4️⃣ After dinner: rate a wine you tried")

    # Reload latest user + wines
    user = load_user_profile(user_id)
    wines_db = load_wines_db()

    # Build options from favorites in profile
    favorite_options = []
    if user.get("favorite_wines"):
        for fw in user["favorite_wines"]:
            wid = fw["wine_id"]
            w_profile = wines_db.get(wid)
            if not w_profile:
                continue
            label = f"{w_profile.get('name')} ({w_profile.get('region')}, {w_profile.get('country')})"
            favorite_options.append((wid, label))

        # Build options from favorites in profile
    options: List[tuple[str, str]] = []

    # 1) Saved favorites
    if user.get("favorite_wines"):
        for fw in user["favorite_wines"]:
            wid = fw["wine_id"]
            w_profile = wines_db.get(wid)
            if not w_profile:
                continue
            label = f"{w_profile.get('name')} ({w_profile.get('region')}, {w_profile.get('country')})"
            options.append((wid, label))

    # 2) Wines from the last menu session (if any)
    last_menu_wines = st.session_state.get("last_menu_wines", [])
    for wid, label in last_menu_wines:
        if wid in wines_db:
            options.append((wid, label))

    # Deduplicate by wine_id (keep first label)
    dedup: Dict[str, str] = {}
    for wid, label in options:
        if wid not in dedup:
            dedup[wid] = label
    favorite_options = list(dedup.items())

    if not favorite_options:
        st.info(
            "You don't have any wines saved in your profile or from a recent menu. "
            "Add wines you like (via text/photo) or run a menu recommendation first."
        )
    else:
        label_to_id = {label: wid for (wid, label) in favorite_options}
        selected_label = st.selectbox(
            "Which wine did you drink?",
            options=list(label_to_id.keys()),
        )
        selected_wine_id = label_to_id[selected_label]

        # Rating: 1–5 with your scale
        rating_label = st.radio(
            "How did you feel about this wine?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1 – Hated it",
                2: "2 – Didn't love it",
                3: "3 – Fine / mediocre",
                4: "4 – Enjoyed it",
                5: "5 – Phenomenal",
            }[x],
            index=3,  # default to 4
        )
        rating_value = rating_label

        context = st.text_input("Where / when did you drink it? (optional)", "")
        notes = st.text_area("Any brief notes? (optional)", "")

        if st.button("💾 Save this rating and update my profile"):
            try:
                user = add_tasting(
                    user_id=user_id,
                    wine_id=selected_wine_id,
                    rating=rating_value,
                    context=context or None,
                    notes=notes or None,
                )
                st.success("Saved your rating and updated your taste profile.")
            except Exception as e:
                st.error(f"Error saving rating: {e}")

if __name__ == "__main__":
    main()
