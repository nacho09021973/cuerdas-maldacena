import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF


PDF_PATH_DEFAULT = Path("data/papers/maldacena_1999.pdf")
OUT_DIR_DEFAULT = Path("data/corpus/m99")

# --- Utils

def page_num(page_id: str) -> int:
    # "M99:p0108" -> 108
    return int(page_id.split(":p")[1])

def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# --- Core build (pages/blocks/paragraphs)

def build_index(pdf_path: Path, out_dir: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF at {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    pages_path = out_dir / "pages.jsonl"
    blocks_path = out_dir / "blocks.jsonl"
    paras_path = out_dir / "paragraphs.jsonl"

    pages_records: List[Dict[str, Any]] = []
    blocks_records: List[Dict[str, Any]] = []
    para_records: List[Dict[str, Any]] = []

    per_page_blocks: Dict[str, List[Dict[str, Any]]] = {}

    # Header/footer filtering (tuned)
    TOP_CUT = 90.0
    BOT_CUT = 70.0

    for i, page in enumerate(doc, start=1):
        page_id = f"M99:p{i:04d}"
        page_h = float(page.rect.height)

        pages_records.append({
            "doc_id": "M99",
            "page": i,
            "page_id": page_id,
            "width": float(page.rect.width),
            "height": float(page.rect.height),
            "text": page.get_text(),
        })

        d = page.get_text("dict")
        block_order = 0
        page_blocks: List[Dict[str, Any]] = []

        for b in d.get("blocks", []):
            if "lines" not in b:
                continue

            # rebuild block text
            lines = []
            for ln in b.get("lines", []):
                spans = [sp.get("text", "") for sp in ln.get("spans", [])]
                lines.append("".join(spans))
            block_text = "\n".join(lines).strip()
            if not block_text:
                continue

            x0, y0, x1, y1 = map(float, b["bbox"])
            t = block_text.strip()

            # micro numeric noise (you validated this is safe)
            w = x1 - x0
            h = y1 - y0
            if t.isdigit() and w <= 10.0 and h <= 14.0:
                continue

            # header/footer bands
            in_header = (y0 < TOP_CUT)
            in_footer = (y1 > (page_h - BOT_CUT))

            # page numbers: digits in lower half => out (robust across layouts)
            if t.isdigit() and y0 > (0.60 * page_h):
                continue

            if in_header or in_footer:
                tl = t.lower()
                if tl.startswith("arxiv:") or ("hep-th/" in tl) or ("arxiv:" in tl):
                    continue
                if len(t) <= 8:
                    continue

            block_order += 1
            rec = {
                "doc_id": "M99",
                "page_id": page_id,
                "block_id": f"{page_id}:b{block_order:03d}",
                "order": block_order,
                "bbox": [x0, y0, x1, y1],
                "text": block_text,
            }
            blocks_records.append(rec)
            page_blocks.append(rec)

        per_page_blocks[page_id] = page_blocks

    # paragraph segmentation (deterministic)
    GAP_Y = 22.0

    def merge_bbox(b1: List[float], b2: List[float]) -> List[float]:
        return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

    for page_id, blks in per_page_blocks.items():
        blks = sorted(blks, key=lambda r: r["order"])

        para_idx = 0
        cur_text_parts: List[str] = []
        cur_bbox: List[float] = []
        cur_block_ids: List[str] = []

        def flush() -> None:
            nonlocal para_idx, cur_text_parts, cur_bbox, cur_block_ids
            if not cur_text_parts:
                return
            text = "\n".join(cur_text_parts).strip()
            if not text:
                cur_text_parts, cur_bbox, cur_block_ids = [], [], []
                return
            para_idx += 1
            para_records.append({
                "doc_id": "M99",
                "page_id": page_id,
                "para_id": f"{page_id}:para{para_idx:03d}",
                "order_in_page": para_idx,
                "bbox": cur_bbox,
                "block_ids": cur_block_ids,
                "text": text,
            })
            cur_text_parts, cur_bbox, cur_block_ids = [], [], []

        prev_bbox: Optional[List[float]] = None
        prev_text: Optional[str] = None

        for b in blks:
            text = b["text"].strip()
            bbox = b["bbox"]

            new_para = False
            if prev_bbox is not None:
                gap = bbox[1] - prev_bbox[3]
                if gap > GAP_Y:
                    new_para = True
                # weak signal: end of sentence + uppercase (only if some gap)
                if (not new_para) and gap > 10:
                    if prev_text and prev_text.rstrip().endswith((".", ":", ";")) and text[:1].isupper():
                        new_para = True

            if new_para:
                flush()

            cur_text_parts.append(text)
            cur_block_ids.append(b["block_id"])
            if not cur_bbox:
                cur_bbox = bbox
            else:
                cur_bbox = merge_bbox(cur_bbox, bbox)

            prev_bbox = bbox
            prev_text = text

        flush()

    write_jsonl(pages_path, pages_records)
    write_jsonl(blocks_path, blocks_records)
    write_jsonl(paras_path, para_records)

    print(f"Indexed {len(doc)} pages")
    print(f"Wrote: {pages_path} ({len(pages_records)} records)")
    print(f"Wrote: {blocks_path} ({len(blocks_records)} records)")
    print(f"Wrote: {paras_path} ({len(para_records)} records)")


# --- Query (Citation Locator basic)

def query_paragraphs(corpus_dir: Path, text: str, max_hits: int, no_front_matter: bool) -> None:
    para_path = corpus_dir / "paragraphs_sections.jsonl"
    if not para_path.exists():
        raise FileNotFoundError(
            f"Missing {para_path}. Generate it first (you already did via the helper script)."
        )

    rx = re.compile(re.escape(text), re.IGNORECASE)

    def is_front_matter(rec: Dict[str, Any]) -> bool:
        if rec.get("section_id") is None:
            return True
        if page_num(rec["page_id"]) <= 6:
            return True
        return False

    hits = 0
    for r in iter_jsonl(para_path):
        if no_front_matter and is_front_matter(r):
            continue

        if rx.search(r["text"]):
            snippet = rx.sub(lambda m: f"<<{m.group(0)}>>", r["text"])
            snippet = " ".join(snippet.split())
            print("\n---")
            print("section_id:", r.get("section_id"))
            print("page_id:", r["page_id"], "para_id:", r["para_id"])
            print("bbox:", r["bbox"])
            print("snippet:", snippet[:260] + ("..." if len(snippet) > 260 else ""))
            hits += 1
            if hits >= max_hits:
                break

    print("\nhits:", hits)


def main():
    parser = argparse.ArgumentParser(prog="m99_index.py")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="Build pages/blocks/paragraphs index from PDF")
    p_build.add_argument("--pdf", type=Path, default=PDF_PATH_DEFAULT)
    p_build.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT)

    p_query = sub.add_parser("query", help="Search paragraphs_sections.jsonl and print citables")
    p_query.add_argument("--corpus", type=Path, default=OUT_DIR_DEFAULT)
    p_query.add_argument("--text", required=True)
    p_query.add_argument("--max", type=int, default=10)
    p_query.add_argument("--no-front-matter", action="store_true", default=False)

    args = parser.parse_args()

    # Backwards-compatible: if no subcommand, behave like build()
    if args.cmd is None:
        build_index(PDF_PATH_DEFAULT, OUT_DIR_DEFAULT)
        return

    if args.cmd == "build":
        build_index(args.pdf, args.out)
        return

    if args.cmd == "query":
        query_paragraphs(args.corpus, args.text, args.max, args.no_front_matter)
        return


if __name__ == "__main__":
    main()