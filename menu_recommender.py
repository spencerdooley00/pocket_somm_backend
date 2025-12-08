import os
from typing import List, Dict, Any, Tuple

import numpy as np

# Adjust these imports based on your actual file names:
from user_profile import build_user_flavor_vector, get_embedding
from wine_recommender import call_gpt41_extract_wines  # <-- this is your existing function

GRAPE_STYLE_HINTS: Dict[str, str] = {
    # Reds
    "nebbiolo": (
        "Typically pale-colored but intensely flavored, with high acidity, high tannin, "
        "red cherry, rose, tar, and often a savory, earthy character."
    ),
    "sangiovese": (
        "Medium-bodied with high acidity, red cherry, tomato, herbs, and a slightly rustic edge."
    ),
    "gamay": (
        "Light-bodied, high-acid red with bright red berry fruit, low tannin, "
        "often juicy and chillable."
    ),
    "pinot noir": (
        "Light to medium-bodied, with red cherry, cranberry, floral notes, "
        "silky tannins and fresh acidity."
    ),
    "cabernet sauvignon": (
        "Full-bodied with dark fruit, firm tannins, and often noticeable oak, "
        "showing blackcurrant, cedar, and spice."
    ),
    "merlot": (
        "Medium to full-bodied with plummy fruit, softer tannins, and a round texture."
    ),
    "syrah": (
        "Medium to full-bodied with dark fruit, pepper, and savory or smoky notes."
    ),
    "agiorgitiko": (
        "Greek red grape, typically medium-bodied with red fruit, spice, and moderate tannin."
    ),
    "xinomavro": (
        "Greek red grape with high acidity and high tannin, often compared to Nebbiolo, "
        "showing red fruit, tomato, and olive notes."
    ),
    "montepulciano": (
        "Italian red grape with dark fruit, medium to full body, and moderate tannin."
    ),
    "tempranillo": (
        "Medium-bodied with red and dark fruit, moderate tannin, and often some oak spice."
    ),

    # Whites
    "assyrtiko": (
        "Greek white grape with high acidity, citrus and saline character, "
        "often mineral and linear."
    ),
    "sauvignon blanc": (
        "Crisp, high-acid white with citrus, green fruit, and herbal notes."
    ),
    "chardonnay": (
        "Can range from lean and mineral to rich and oaky; often medium to full-bodied with apple, citrus, "
        "and sometimes butter or vanilla if oaked."
    ),
    "malagousia": (
        "Greek white grape with aromatic profile, stone fruit, citrus, and floral notes, "
        "medium body and fresh acidity."
    ),
    "riesling": (
        "High acidity, aromatic, with citrus, stone fruit, and sometimes petrol; can be dry or sweet."
    ),

    # You can extend this mapping over time.
}

# =========================
# BUILD EMBEDDING TEXT FOR MENU WINES
# =========================
def build_menu_wine_embedding_text(w: Dict[str, Any]) -> str:
    """
    Turn a structured menu wine entry into a descriptive text string.

    Fields expected:
      - name (str)
      - producer (str | None)
      - region (str | None)
      - country (str | None)
      - color (str | None)  # red, white, rosé, orange, sparkling, other
      - grapes (list[str])
      - vintage (str | None)
      - section (str | None)  # 'RED', 'WHITE', etc.

    We *do not* call GPT here; we use structured info plus grape-style heuristics
    to make a richer description for embeddings.
    """
    name = w.get("name") or "Unknown wine"
    producer = w.get("producer")
    region = w.get("region")
    country = w.get("country")
    color = (w.get("color") or "wine").lower()
    grapes = w.get("grapes") or []
    vintage = w.get("vintage")
    section = w.get("section")

    parts: List[str] = []

    # Base identity
    if producer and producer.lower() not in name.lower():
        parts.append(f"{name} by {producer} is a")
    else:
        parts.append(f"{name} is a")

    # Color + type
    if color in ["red", "white", "rosé", "rose", "orange", "sparkling"]:
        parts.append(f" {color} wine")
    else:
        parts.append(" wine")

    # Region / country
    loc_bits = []
    if region:
        loc_bits.append(region)
    if country:
        loc_bits.append(country)
    if loc_bits:
        parts.append(" from " + ", ".join(loc_bits))

    # Vintage
    if vintage:
        parts.append(f", vintage {vintage}")

    # Section hint (if not just basic color)
    if section and section.upper() not in ["RED", "WHITE", "ROSÉ", "ROSE"]:
        parts.append(f". It is listed in the {section} section of the menu")

    # Grapes + style hints
    style_snippets: List[str] = []
    grape_names_clean: List[str] = []

    for g in grapes:
        g_stripped = (g or "").strip()
        if not g_stripped:
            continue
        grape_names_clean.append(g_stripped)
        key = g_stripped.lower()
        # normalize simple variants
        key = key.replace("é", "e")

        hint = GRAPE_STYLE_HINTS.get(key)
        if hint:
            style_snippets.append(f"As a wine made from {g_stripped}, it typically shows {hint}")

    if grape_names_clean:
        parts.append(f". It is made from {', '.join(grape_names_clean)}")

    # Color-based generic cues if no grape hints
    if not style_snippets:
        if color == "red":
            style_snippets.append(
                "It is a red wine, likely with red or dark fruit, tannin, and varying levels of oak and body."
            )
        elif color == "white":
            style_snippets.append(
                "It is a white wine, likely with citrus, stone fruit, or floral notes and varying acidity and body."
            )
        elif color in ["rosé", "rose"]:
            style_snippets.append(
                "It is a rosé wine, typically dry, refreshing, and red-fruited."
            )
        elif color == "sparkling":
            style_snippets.append(
                "It is a sparkling wine with bubbles and higher acidity, suitable as an aperitif or with food."
            )
        elif color == "orange":
            style_snippets.append(
                "It is an orange, or skin-contact white wine, with some tannic grip and tea-like, dried-fruit notes."
            )

    if style_snippets:
        parts.append(". " + " ".join(style_snippets))

    description = " ".join(parts)
    return description

# =========================
# SCORE MENU FOR A USER
# =========================

def score_menu_for_user(
    user_vec: np.ndarray,
    menu_wines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Given a user flavor vector and a list of menu wines (from GPT extraction),
    compute cosine similarity between the user vector and each menu wine embedding.

    Returns:
      A sorted list of dicts with:
        - "wine": original wine dict
        - "score": float similarity
        - "embedding_text": text used for embedding
    """
    results = []

    for w in menu_wines:
        emb_text = build_menu_wine_embedding_text(w)
        emb = get_embedding(emb_text)  # normalized

        score = float(np.dot(user_vec, emb))  # cosine similarity

        results.append(
            {
                "wine": w,
                "score": score,
                "embedding_text": emb_text,
            }
        )

    # sort descending by similarity
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def pretty_print_ranked_menu(results: List[Dict[str, Any]], top_k: int = 10) -> None:
    """
    Display the top_k wines in a user-friendly way.
    """
    print(f"\n================ RECOMMENDED WINES (top {top_k}) ================\n")
    for i, item in enumerate(results[:top_k], start=1):
        w = item["wine"]
        score = item["score"]

        name = w.get("name")
        vintage = w.get("vintage")
        section = w.get("section")
        color = w.get("color")
        region = w.get("region")
        country = w.get("country")
        grapes = w.get("grapes") or []
        price = w.get("price_string")

        header = f"{i:2d}. {name}"
        if vintage:
            header += f" ({vintage})"
        if section:
            header += f"  [{section}]"

        header += f"  — similarity: {score:.3f}"
        print(header)

        detail_bits = []
        if color:
            detail_bits.append(color)
        if region:
            detail_bits.append(region)
        if country:
            detail_bits.append(country)
        if detail_bits:
            print("    " + ", ".join(detail_bits))

        if grapes:
            print(f"    Grapes: {', '.join(grapes)}")
        if price:
            print(f"    Price:  {price}")

        print()


# =========================
# SIMPLE END-TO-END TEST
# =========================

def main():
    """
    End-to-end test:
      1) Ask user for favorite wines
      2) Build user flavor vector
      3) Ask for menu PDF path
      4) Extract structured wines via call_gpt41_extract_wines
      5) Score and print ranked recommendations
    """
    # 1) collect favorites (reuse your interactive helper or inline)
    from user_profile import _interactive_collect_favorites  # already defined there

    favorites = _interactive_collect_favorites(max_wines=5)
    if not favorites:
        print("No wines entered. Exiting.")
        return

    user_vec, profiles = build_user_flavor_vector(favorites)
    print("\n[INFO] Built user flavor vector.")

    # 2) ask for menu PDF
    menu_path = input("\nEnter path to menu PDF (e.g. menus/BV_MENU_Wine_June30.2025.pdf): ").strip()
    if not os.path.exists(menu_path):
        print(f"Menu file not found: {menu_path}")
        return

    # 3) extract wines from menu via GPT
    print(f"[INFO] Extracting wines from menu: {menu_path}")
    menu_wines = call_gpt41_extract_wines(menu_path)
    print(f"[INFO] Extracted {len(menu_wines)} wines from menu.")

    # 4) score menu
    results = score_menu_for_user(user_vec, menu_wines)

    # 5) pretty print
    pretty_print_ranked_menu(results, top_k=10)


if __name__ == "__main__":
    main()
