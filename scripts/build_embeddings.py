"""
Section-aware embedding pipeline for the Methods Assistant.

Reads a marker-produced Markdown file (with LaTeX equations preserved and
page anchors embedded as <span id="page-N-M"></span>), splits on heading
boundaries into coherent chunks, and embeds each chunk with all-MiniLM-L6-v2.

Output JSON schema:
    {
      "model": "...", "dim": 384, "quantization": "int8", "quant_scale": 127,
      "source": {...},
      "chunks": [
        { "id": 0, "text": "...markdown with $\\LaTeX$...",
          "section_title": "Yee grid in 3D",
          "section_path": "4. FDTD › 4.3 3D problems › 4.3.1 Yee grid in 3D",
          "page_start": 61, "page_end": 61 },
        ...
      ],
      "vectors": [[int8, ...], ...]
    }

Usage:
    python build_embeddings_md.py <markdown_path> <output_json>
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_WORDS = 380
OVERLAP_WORDS = 60

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SPAN_RE = re.compile(r'<span id="page-(\d+)-\d+"></span>')
TAG_RE = re.compile(r"<[^>]+>")
BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*")


def clean_title(raw: str) -> tuple[str, int | None]:
    m = SPAN_RE.search(raw)
    page = int(m.group(1)) + 1 if m else None  # 0-indexed → 1-indexed for display
    s = TAG_RE.sub("", raw)
    s = BOLD_RE.sub(r"\1", s)
    s = ITALIC_RE.sub(r"\1", s)
    return s.strip(), page


def parse_sections(md: str):
    """Yield (title, level, body, page) for each heading encountered."""
    lines = md.splitlines()
    cur_title: str | None = None
    cur_level = 0
    cur_page: int | None = None
    cur_body: list[str] = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if cur_title is not None:
                yield cur_title, cur_level, "\n".join(cur_body).strip(), cur_page
            title, page = clean_title(m.group(2))
            cur_title = title
            cur_level = len(m.group(1))
            cur_page = page
            cur_body = []
        else:
            if cur_title is not None:
                cur_body.append(line)

    if cur_title is not None:
        yield cur_title, cur_level, "\n".join(cur_body).strip(), cur_page


def strip_toc_and_title(sections):
    """Skip the top-level book title and any section with no body until we hit
    the first real chapter (level == 1 with a body starting with '0.' or a number)."""
    out = list(sections)
    # Skip sections until we hit the first one whose title starts with a digit followed by '.'
    start = 0
    for i, (title, level, body, _page) in enumerate(out):
        if re.match(r"^\s*\d+\.", title) and body:
            start = i
            break
    return out[start:]


def build_breadcrumbs(sections):
    """Assign a section_path breadcrumb (hierarchical) to each section."""
    path_stack: list[tuple[int, str]] = []
    result = []
    for title, level, body, page in sections:
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, title))
        crumbs = [t for (_l, t) in path_stack]
        section_path = " › ".join(crumbs)
        result.append(
            {
                "title": title,
                "level": level,
                "section_path": section_path,
                "body": body,
                "page": page,
            }
        )
    return result


def strip_markdown_for_wordcount(text: str) -> list[str]:
    # Remove display math and inline math for word counting purposes so a long
    # equation doesn't dominate. But keep in the final chunk body.
    t = re.sub(r"\$\$.*?\$\$", " [math] ", text, flags=re.DOTALL)
    t = re.sub(r"\$[^$]+\$", " [math] ", t)
    return t.split()


def split_body_if_needed(body: str) -> list[str]:
    """Split a long section body into overlapping windows on word boundaries."""
    words = body.split()
    if len(words) <= MAX_WORDS:
        return [body] if body.strip() else []
    windows = []
    step = MAX_WORDS - OVERLAP_WORDS
    i = 0
    while i < len(words):
        j = min(i + MAX_WORDS, len(words))
        windows.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i += step
    return windows


def build_chunks(sections):
    out = []
    cid = 0
    for s in sections:
        body = s["body"]
        if not body.strip():
            continue
        # Count meaningful words (ignore math-heavy sections being over-split)
        n_words = len(body.split())
        if n_words < 10:
            # very short — skip or merge? for now, keep if it has content
            continue
        pieces = split_body_if_needed(body)
        for piece in pieces:
            out.append(
                {
                    "id": cid,
                    "text": piece,
                    "section_title": s["title"],
                    "section_path": s["section_path"],
                    "page_start": s["page"],
                    "page_end": s["page"],
                }
            )
            cid += 1
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: build_embeddings_md.py <md_path> <output_json>", file=sys.stderr)
        return 2

    md_path = Path(sys.argv[1]).expanduser()
    out_path = Path(sys.argv[2]).expanduser()
    if not md_path.exists():
        print(f"error: {md_path} not found", file=sys.stderr)
        return 1

    md = md_path.read_text(encoding="utf-8")
    print(f"[1/5] parsing {md_path.name}")
    raw_sections = list(parse_sections(md))
    print(f"      {len(raw_sections)} raw headings")

    sections = strip_toc_and_title(raw_sections)
    print(f"      {len(sections)} after stripping title/TOC")

    sections = build_breadcrumbs(sections)

    chunks = build_chunks(sections)
    print(f"[2/5] {len(chunks)} chunks (max {MAX_WORDS} words, overlap {OVERLAP_WORDS})")

    # Embedding text: prepend the section path so the semantic vector captures the hierarchy
    embed_texts = [f"{c['section_path']}\n\n{c['text']}" for c in chunks]

    print(f"[3/5] loading model {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID)

    print(f"[4/5] embedding {len(chunks)} chunks")
    vecs = model.encode(
        embed_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    q = np.round(vecs * 127).clip(-127, 127).astype(np.int8)

    print(f"[5/5] writing {out_path}")
    out = {
        "model": MODEL_ID,
        "dim": int(vecs.shape[1]),
        "quantization": "int8",
        "quant_scale": 127,
        "source": {
            "id": "cpho-notes-01",
            "title": "Internal course notes",
        },
        "chunks": chunks,
        "vectors": q.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False))
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
