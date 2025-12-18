#!/usr/bin/env python3
"""
M99 Agent Wrapper
-----------------
Turns a natural language question into structured evidence
from the Maldacena 1999 corpus using m99_index.py.

This script does NOT answer questions.
It only retrieves citable evidence.
"""

import argparse
import subprocess
import sys
import json
import shlex


M99_QUERY_CMD = [
    sys.executable,
    "tools/m99_index.py",
    "query",
    "--no-front-matter",
    "--jsonl",
]


def run_query(query: str, max_hits: int):
    cmd = M99_QUERY_CMD + ["--text", query, "--max", str(max_hits)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())

    hits = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        hits.append(json.loads(line))

    return hits


def main():
    parser = argparse.ArgumentParser(description="M99 citation agent wrapper")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--max", type=int, default=10, help="Max hits per query")

    args = parser.parse_args()

    question = args.question.strip()
    if not question:
        raise ValueError("Empty question")

    # Very conservative query strategy:
    # extract key noun phrases by hand (no ML, no guessing)
    # For now: use the full question AND a few obvious substrings
    queries = {question}

    # crude heuristic expansions (safe, transparent)
    lowered = question.lower()
    if "wilson" in lowered:
        queries.add("Wilson loop")
    if "correlator" in lowered or "correlation" in lowered:
        queries.add("correlation function")
    if "finite temperature" in lowered:
        queries.add("finite temperature")

    all_hits = []
    for q in queries:
        hits = run_query(q, args.max)
        all_hits.extend(hits)

    # Deduplicate by para_id
    seen = set()
    unique_hits = []
    for h in all_hits:
        pid = h.get("para_id")
        if pid and pid not in seen:
            seen.add(pid)
            unique_hits.append(h)

    # Output: pure JSON (for GPT-5.2 consumption)
    print(json.dumps({
        "question": question,
        "queries_used": sorted(list(queries)),
        "hits": unique_hits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
