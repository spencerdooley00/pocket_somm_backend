# taste_survey.py

from typing import List, Dict, Any, Tuple
import numpy as np

from user_profile import get_embedding  # reuse your existing embedding helper


# =========================
# 1. STYLE DESCRIPTIONS
# =========================

STYLE_PROMPTS = {
    "light_fruity_red": (
        "light-bodied, fruity red wine with bright red berries, "
        "low tannin, high acidity, often served slightly chilled, "
        "similar to Beaujolais or Pinot Noir."
    ),
    "medium_red": (
        "medium-bodied red wine with balanced fruit and earth, "
        "moderate tannin and acidity, like Chianti, Rioja, or Cabernet Franc."
    ),
    "bold_red": (
        "full-bodied, rich red wine with dark fruit, higher tannin, "
        "riper style and noticeable oak, similar to Napa Cabernet, Malbec or Syrah."
    ),
    "crisp_white": (
        "light-bodied, crisp white wine with high acidity, citrus and mineral notes, "
        "like Sauvignon Blanc, Assyrtiko or dry Riesling."
    ),
    "rich_white": (
        "fuller-bodied white wine with softer acidity, more texture and sometimes oak, "
        "like Chardonnay or Viognier."
    ),
    "rose": (
        "dry rosé with red berry fruit, light body and refreshing acidity, "
        "often similar in feel to a light red but more delicate."
    ),
    "sparkling": (
        "sparkling wine with bubbles, high acidity, and a refreshing, lively profile, "
        "like Champagne, Cava or Prosecco."
    ),
    "orange": (
        "orange or skin-contact white wine with tannic grip, tea-like notes, "
        "and often savory, herbal or dried fruit character."
    ),
    "sweet": (
        "noticeably sweet wine with ripe fruit, lower perceived acidity, "
        "and dessert-like character, like Moscato or late-harvest Riesling."
    ),
}

PREFERENCE_PROMPTS = {
    "tannin_pref": {
        "low": (
            "prefers wines with very soft, low tannin, smooth texture and minimal grip."
        ),
        "medium": (
            "comfortable with moderate tannin and some grip, enjoying some structure but not aggressive."
        ),
        "high": (
            "enjoys firm, structured wines with clearly noticeable tannin and grip."
        ),
    },
    "acidity_pref": {
        "low": (
            "prefers wines with softer, rounder acidity and a smoother mouthfeel."
        ),
        "medium": (
            "likes balanced acidity, where wines are fresh but not sharp."
        ),
        "high": (
            "enjoys high-acid wines that feel bright, zesty and mouthwatering."
        ),
    },
    "oak_pref": {
        "low": (
            "prefers wines with little to no oak influence, focusing on pure fruit and freshness."
        ),
        "medium": (
            "likes some subtle oak character, adding a bit of spice or roundness without dominating."
        ),
        "high": (
            "enjoys clear oak influence with notes of vanilla, toast, baking spices and a richer texture."
        ),
    },
    "adventure_pref": {
        "low": (
            "prefers classic, familiar wine styles and does not want strong funk or unusual flavors."
        ),
        "medium": (
            "is open to trying new styles and lesser-known regions while still generally liking clean wines."
        ),
        "high": (
            "is excited by unusual, funky or experimental wines, including natural, orange and wild styles."
        ),
    },
}


# =========================
# 2. BUILD STYLE EMBEDDING FROM SURVEY
# =========================
def build_style_embedding_from_survey(survey_answers: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a user's survey answers into a flavor-space embedding.

    Expected survey_answers keys:
      - "favorite_styles": List[str] subset of STYLE_PROMPTS keys
      - "tannin_pref": "low" | "medium" | "high"
      - "acidity_pref": "low" | "medium" | "high"
      - "oak_pref": "low" | "medium" | "high"
      - "adventure_pref": "low" | "medium" | "high"

    Strategy:
      - Map each choice to descriptive sentences.
      - Build one synthetic "style summary" paragraph.
      - Embed all texts and average, then normalize.
    """

    favorite_styles: List[str] = survey_answers.get("favorite_styles", []) or []
    tannin_pref: str = survey_answers.get("tannin_pref", "medium")
    acidity_pref: str = survey_answers.get("acidity_pref", "medium")
    oak_pref: str = survey_answers.get("oak_pref", "medium")
    adventure_pref: str = survey_answers.get("adventure_pref", "medium")

    texts: List[str] = []

    # --- A. Favorite style descriptions ---
    STYLE_DESCRIPTIONS: Dict[str, str] = {
        "light_fruity_red": (
            "They enjoy light-bodied, juicy red wines such as Beaujolais or lighter Pinot Noir, "
            "with bright red fruit, high acidity, and low tannin. Often chillable and easy-drinking."
        ),
        "medium_red": (
            "They enjoy medium-bodied reds like Chianti, Rioja, or Cabernet Franc, "
            "with a balance of fruit, acidity, and structure."
        ),
        "bold_red": (
            "They enjoy fuller-bodied reds like Napa Cabernet, Malbec, or Syrah, "
            "with darker fruit, more tannin, and often some oak influence."
        ),
        "crisp_white": (
            "They like crisp, refreshing whites such as Sauvignon Blanc or Assyrtiko, "
            "with bright acidity and clean, citrusy or mineral profiles."
        ),
        "rich_white": (
            "They enjoy richer, more textured whites such as Chardonnay or Viognier, "
            "with more body, softer acidity, and sometimes oak or creamy texture."
        ),
        "rose": (
            "They enjoy rosé wines, typically dry, refreshing, and red-fruited."
        ),
        "sparkling": (
            "They enjoy sparkling wines like Champagne, Prosecco, or Cava, "
            "with lively bubbles and high acidity."
        ),
        "orange": (
            "They enjoy orange or skin-contact white wines, with tannic texture, "
            "dried fruit, tea, and herbal notes."
        ),
        "sweet": (
            "They enjoy sweet wines such as Moscato, Sauternes, or sweet Riesling, "
            "with noticeable residual sugar balancing the acidity."
        ),
    }

    for key in favorite_styles:
        desc = STYLE_DESCRIPTIONS.get(key)
        if desc:
            texts.append(desc)

    # --- B. Structural preferences ---
    def pref_sentence(name: str, value: str, low: str, med: str, high: str) -> str:
        if value == "low":
            return low
        elif value == "high":
            return high
        else:
            return med

    tannin_sentence = pref_sentence(
        "tannin",
        tannin_pref,
        "They prefer wines with very smooth, low tannin, without much grip.",
        "They like firm, structured wines with noticeable tannin.",
        "They enjoy clearly tannic wines with a lot of grip and structure.",
    )

    acidity_sentence = pref_sentence(
        "acidity",
        acidity_pref,
        "They prefer softer, rounder wines with lower perceived acidity.",
        "They like a balanced, medium level of acidity.",
        "They love bright, zippy wines with high acidity and freshness.",
    )

    oak_sentence = pref_sentence(
        "oak",
        oak_pref,
        "They prefer little to no oak influence, favoring stainless steel or neutral vessels.",
        "They enjoy some subtle oak influence, adding gentle spice and texture.",
        "They like clearly oaky wines with vanilla, toast, and baking-spice notes.",
    )

    adventure_sentence = pref_sentence(
        "adventure",
        adventure_pref,
        "They prefer safe, familiar styles and classic regions over experimental wines.",
        "They are open to some new grapes and regions but still like a familiar core.",
        "They are adventurous and happy to explore unusual grapes, regions, and styles.",
    )

    texts.extend([tannin_sentence, acidity_sentence, oak_sentence, adventure_sentence])

    # --- C. Synthetic summary paragraph ---
    # Build a compact narrative of their style.
    style_labels_readable = []
    for key in favorite_styles:
        label = key.replace("_", " ")
        style_labels_readable.append(label)

    if style_labels_readable:
        styles_str = ", ".join(style_labels_readable)
    else:
        styles_str = "a range of wine styles"

    summary = (
        f"This drinker tends to enjoy {styles_str}. "
        f"{tannin_sentence} {acidity_sentence} {oak_sentence} {adventure_sentence} "
        "Overall they are looking for wines that match this profile in body, acidity, tannin, oak, and adventurousness."
    )

    texts.append(summary)

    # --- D. Embed and average ---
    embedding_vectors: List[np.ndarray] = []
    for t in texts:
        emb = get_embedding(t)
        embedding_vectors.append(emb)

    if not embedding_vectors:
        # Fall back to a neutral vector
        style_vec = np.zeros((3072,), dtype=np.float32)
    else:
        mat = np.stack(embedding_vectors, axis=0)
        style_vec = mat.mean(axis=0)
        norm = np.linalg.norm(style_vec)
        if norm > 0:
            style_vec = style_vec / norm

    debug_info = {
        "texts": texts,
        "survey_answers": survey_answers,
    }

    return style_vec, debug_info

# =========================
# 3. SIMPLE CLI TEST
# =========================

def _interactive_survey() -> Dict[str, Any]:
    """
    Very basic CLI version of the survey for testing.
    In the real app this will be UI, not input() prompts.
    """
    print("Answer a few quick questions about your taste.\n")

    print("Q1: What kinds of wines do you usually enjoy? (comma-separated)")
    print("Options: light_fruity_red, medium_red, bold_red, crisp_white, rich_white, rose, sparkling, orange, sweet")
    fav = input("Your choices: ").strip()
    favorite_styles = [x.strip() for x in fav.split(",") if x.strip()]

    def ask_pref(q, options):
        print(f"\n{q} ({'/'.join(options)})")
        val = input("Your choice: ").strip().lower()
        while val not in options:
            print(f"Please choose one of: {', '.join(options)}")
            val = input("Your choice: ").strip().lower()
        return val

    tannin_pref = ask_pref("Q2: How do you feel about tannins? (smooth vs grippy)", ["low", "medium", "high"])
    acidity_pref = ask_pref("Q3: How do you feel about acidity? (soft vs zippy)", ["low", "medium", "high"])
    oak_pref = ask_pref("Q4: How much oak do you like?", ["low", "medium", "high"])
    adventure_pref = ask_pref("Q5: How adventurous are you?", ["low", "medium", "high"])

    return {
        "favorite_styles": favorite_styles,
        "tannin_pref": tannin_pref,
        "acidity_pref": acidity_pref,
        "oak_pref": oak_pref,
        "adventure_pref": adventure_pref,
    }


if __name__ == "__main__":
    answers = _interactive_survey()
    style_vec, debug = build_style_embedding_from_survey(answers)

    print("\n================ STYLE VECTOR (first 10 dims) ================")
    print(style_vec[:10])
    print(f"Vector length: {len(style_vec)}\n")

    print("================ TEXTS USED FOR EMBEDDING ================\n")
    for t in debug["texts"]:
        print("-", t)
