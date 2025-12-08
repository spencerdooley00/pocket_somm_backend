import os
import sys
import base64
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


# =========================
# 1. CONFIG / CLIENT
# =========================

load_dotenv()  # loads OPENAI_API_KEY from .env

client = OpenAI()  # uses OPENAI_API_KEY env var


# =========================
# 2. HELPERS
# =========================

def encode_pdf_to_data_url(pdf_path: str) -> str:
    """
    Read a PDF file and return a data: URL string suitable for input_file/file_data.
    """
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def call_gpt41_extract_wines(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Use GPT-4.1 via the Responses API to extract wines from a menu PDF.

    Returns:
        A list of dicts, each like:
        {
          "name": str,
          "producer": str | null,
          "region": str | null,
          "country": str | null,
          "color": str | null,
          "grapes": [str],
          "vintage": str | null,
          "price_string": str | null,
          "section": str | null
        }
    """
    file_data_url = encode_pdf_to_data_url(pdf_path)
    filename = os.path.basename(pdf_path)

    system_prompt = (
        "You are an expert sommelier and structured data extractor. "
        "You are given a PDF of a restaurant wine list. "
        "Extract every distinct wine listed and return ONLY valid JSON. "
        "Infer fields when obvious, otherwise use null.\n\n"
        "For EACH wine, return an object with:\n"
        "- name: full wine name including cuvée (string)\n"
        "- producer: producer or estate (string or null)\n"
        "- region: region/appellation, e.g. 'Naoussa', 'PGI Tyrnavos' (string or null)\n"
        "- country: country (e.g. 'Greece') (string or null)\n"
        "- color: one of ['red','white','rosé','orange','sparkling','other'] (string or null)\n"
        "- grapes: array of grape variety names (e.g. ['Xinomavro','Assyrtiko'])\n"
        "- vintage: vintage year as string (e.g. '2021', 'NV') if shown, else null\n"
        "- price_string: the exact price text as shown on the menu, e.g. '20 / 78', '65', '8 Glass / 24 500ml' (string or null)\n"
        "- section: section header on the list if present, e.g. 'RED', 'WHITE', 'FIZZY RED', 'HOUSE WINE' (string or null)\n\n"
        "Important rules:\n"
        "- If two prices (glass / bottle) are shown, keep the whole pattern in price_string.\n"
        "- Do NOT include beers.\n"
        "- Do NOT include non-alcoholic items.\n"
        "- If there is both a house wine section AND specific bottled wines, include all of them.\n"
        "- If a line is clearly the continuation of the previous wine (e.g. just region/vintage/price), merge it into the same wine.\n"
        "- Return ONLY a JSON array, no commentary, no explanation."
    )

    # Responses API call
    response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": system_prompt,
                },
                {
                    "type": "input_file",
                    "filename": filename,
                    "file_data": file_data_url,
                },
            ],
        }
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "wine_list",
            "schema": {
                "type": "object",
                "properties": {
                    "wines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "producer": {"type": ["string", "null"]},
                                "region": {"type": ["string", "null"]},
                                "country": {"type": ["string", "null"]},
                                "color": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        "red",
                                        "white",
                                        "rosé",
                                        "rose",
                                        "orange",
                                        "sparkling",
                                        "other",
                                        None
                                    ],
                                },
                                "grapes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "vintage": {"type": ["string", "null"]},
                                "price_string": {"type": ["string", "null"]},
                                "section": {"type": ["string", "null"]},
                            },
                            "required": ["name", "grapes"],
                        },
                    }
                },
                "required": ["wines"],
            },
            "strict": False,
        }
    },
    temperature=0.1,
)


    # The Responses API returns an object with .output
    # We want to pull out the text from the first output block.
    # Structure: response.output is a list of "output choices"
    # Each has .content, which is a list of parts. We want type=="output_text".
    if not getattr(response, "output", None):
        raise RuntimeError("No output returned from GPT-4.1")

    texts: List[str] = []
    for out in response.output:
        for part in out.content:
            if part.type == "output_text":
                texts.append(part.text)

    if not texts:
        raise RuntimeError("No output_text content found in response")

    raw_json_str = texts[0].strip()

    # Try to parse JSON
    raw_json_str = texts[0].strip()
    root = json.loads(raw_json_str)

    if not isinstance(root, dict) or "wines" not in root:
        raise RuntimeError("Model did not return expected { 'wines': [...] } structure")

    return root["wines"]

def pretty_print_wines(wines: List[Dict[str, Any]]) -> None:
    """
    Print a human-readable summary of extracted wines.
    """
    for i, w in enumerate(wines, start=1):
        name = w.get("name")
        vintage = w.get("vintage")
        section = w.get("section")
        price = w.get("price_string")
        color = w.get("color")
        region = w.get("region")
        grapes = w.get("grapes") or []

        header = f"{i:2d}. {name}"
        if vintage:
            header += f" ({vintage})"
        if section:
            header += f"  [{section}]"
        print(header)

        if region or color:
            print(f"    Region: {region} | Color: {color}")
        if grapes:
            print(f"    Grapes: {', '.join(grapes)}")
        if price:
            print(f"    Price:  {price}")
        print()


# =========================
# 3. MAIN
# =========================

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_wines_from_pdf_gpt.py /path/to/menu.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    wines = call_gpt41_extract_wines(pdf_path)

    # Print raw JSON (for saving to file / debugging)
    print("=============== RAW JSON ===============")
    print(json.dumps(wines, indent=2, ensure_ascii=False))

    print("\n=============== PRETTY VIEW ===============\n")
    pretty_print_wines(wines)


if __name__ == "__main__":
    main()
