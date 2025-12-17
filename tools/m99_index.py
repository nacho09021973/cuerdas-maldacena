import fitz  # PyMuPDF
import json
from pathlib import Path

PDF_PATH = Path("data/papers/maldacena_1999.pdf")
OUT_DIR = Path("data/corpus/m99")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing PDF at {PDF_PATH}")

    doc = fitz.open(PDF_PATH)

    pages_path = OUT_DIR / "pages.jsonl"
    blocks_path = OUT_DIR / "blocks.jsonl"

    pages_records = []
    blocks_records = []

    for i, page in enumerate(doc, start=1):
        page_id = f"M99:p{i:04d}"

        # Página (texto plano)
        pages_records.append({
            "doc_id": "M99",
            "page": i,
            "page_id": page_id,
            "width": float(page.rect.width),
            "height": float(page.rect.height),
            "text": page.get_text()
        })

        # Bloques con layout (bbox)
        d = page.get_text("dict")
        block_order = 0
        for b in d.get("blocks", []):
            # descartamos blocks no-texto (imágenes)
            if "lines" not in b:
                continue

            # reconstruimos texto del bloque
            lines = []
            for ln in b.get("lines", []):
                spans = [sp.get("text", "") for sp in ln.get("spans", [])]
                lines.append("".join(spans))
            block_text = "\n".join(lines).strip()
            if not block_text:
                continue

            block_order += 1
            x0, y0, x1, y1 = b["bbox"]

            blocks_records.append({
                "doc_id": "M99",
                "page_id": page_id,
                "block_id": f"{page_id}:b{block_order:03d}",
                "order": block_order,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": block_text
            })

    write_jsonl(pages_path, pages_records)
    write_jsonl(blocks_path, blocks_records)

    print(f"Indexed {len(doc)} pages")
    print(f"Wrote: {pages_path} ({len(pages_records)} records)")
    print(f"Wrote: {blocks_path} ({len(blocks_records)} records)")

if __name__ == "__main__":
    main()
