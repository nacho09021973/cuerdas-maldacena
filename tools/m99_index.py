import fitz  # PyMuPDF
import json
from pathlib import Path

PDF_PATH = Path("data/papers/maldacena_1999.pdf")
OUT_DIR = Path("data/corpus/m99")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    doc = fitz.open(PDF_PATH)
    pages_out = OUT_DIR / "pages.jsonl"

    with pages_out.open("w", encoding="utf-8") as f:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            record = {
                "doc_id": "M99",
                "page": i,
                "page_id": f"M99:p{i:04d}",
                "text": text
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Indexed {len(doc)} pages")

if __name__ == "__main__":
    main()
