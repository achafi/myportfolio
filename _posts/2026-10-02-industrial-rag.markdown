---
title : "Industrial engineering knowledge assistant (P&ID OCR and RAG)"
date : 2026-10-02
categories : [agentic-ai]
tags : [RAG, OCR, GenAI, Streamlit, Python]
header :
  image : ""
excerpt : "RAG, OCR, P&ID, GenAI"
---
[Source code](https://github.com/achafi/industrial_rag)

# Industrial engineering knowledge assistant
*The objective of this project is to build an assistant that connects the documents an industrial plant actually runs on — P&ID drawings, equipment manuals, procedures, a tag register and maintenance history — so an engineer can ask where `P4711` is and get the answer with evidence. The current milestone goes deep on the hardest input type first: P&ID drawings, local OCR, region classification and exact tag search, with RAG over the corpus layered on top.*

## Introduction

P&ID drawings are where industrial knowledge goes to die: a dense vector drawing with hundreds of small labels, rotated text, equipment tables and title blocks, none of which is machine-readable in any straightforward way. Before you can answer questions *about* a drawing, you have to be able to read it.

This project does that entirely locally — no cloud OCR API, no external service — and then builds the retrieval layer on top: a FastAPI ingestion and vector-search backend over a true/noisy document corpus, plus a Streamlit workspace that switches between drawing tools and an agentic RAG chat.

## P&ID OCR pipeline

The pipeline starts from `data/p&ids/reference_pid.svg` (SVG, PDF and PNG branches are all supported, up to 25 MB):

1. **Render** with CairoSVG at a configurable width, preserving the source aspect ratio.
2. **Tile** the drawing and run RapidOCR (ONNX, bundled models, CPU) over overlapping tiles at **0°, 90° and 270°** — the rotated passes are what recover vertical text, with coordinates mapped back to the original page.
3. **Select** detections with a heuristic ranking: regions away from internal tile boundaries first, then readings above the review threshold, then longer text extent, then confidence. Compatible overlapping readings from different tiles are suppressed, with the discarded alternatives kept for audit.
4. **Classify** every retained region as `process_drawing`, `equipment_data_table`, `title_block` or `drawing_border` using editable, normalized layout zones — explicit geometry, not an automatic table detector.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/industrial_rag/reference_pid.png" alt="Clean high-resolution rendering of the reference P&ID drawing">
*Fig. 1: The clean high-resolution rendering of the reference P&ID — the input to the OCR pipeline.*

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/industrial_rag/reference_pid_ocr_annotated.png" alt="P&ID with OCR detection boxes and IDs annotated">
*Fig. 2: The same drawing with OCR detection boxes and IDs annotated; orange marks low-confidence regions or regions touching internal tile boundaries, gray marks excluded border readings.*

Confidence is RapidOCR's recognition score in [0, 1], not a guarantee of correctness: readings below 0.80 are flagged for review but kept in the export. Every output — JSON with geometry and provenance, CSV summaries, searchable exports, annotated PNG — is written to `outputs/`.

## Tag search and relevant-region retrieval

Search is **contains** by default: `PI4712` finds `PI4712.01` even when the prefix and number arrived as separate OCR boxes, and prefix-only or numeric-fragment queries work too. Exact mode requires whole-tag boundaries, so `P4711` matches but `P47110` does not.

Selecting a match retrieves its **contextual crops**: process-drawing matches get a seed window padded by 5% of page width/height, expanded once to include complete bounds with an 8-pixel margin; configured equipment-table rectangles use center-based selection so neighbouring tables do not bleed in. The match is outlined in red and nearby text in blue, and each result carries its selection method, region ID, confidence and nearby OCR.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/industrial_rag/search_drawing_crop.png" alt="Highlighted crop of pump P4711 on the process drawing">
*Fig. 3: Search result for `P4711` on the process drawing — matched region in red, surrounding labels in blue.*

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/industrial_rag/search_table_crop.png" alt="Highlighted crop of the equipment data table entry for P4711">
*Fig. 4: The same query returns the pump's entry in the lower-left equipment specification table — the same tag, two regions, one report.*

Each search downloads a ZIP with the JSON report and all crops. Searches, filters and downloads never re-run OCR; changing the file or processing settings clears stale results.

## Grouped instrument tags

`pid_instruments.py` builds a provisional inventory by grouping standalone prefix/number OCR boxes that are close and aligned — a number directly below a prefix or immediately to its right: `PT` + `133035` → `PT133035`, `PI` + `4712.01` → `PI4712.01`. The editable `INSTRUMENT_PREFIXES` set defines eligibility, geometry scales with text height, competing partners stay visible with `ambiguous=True`, and original detections are never merged or overwritten. The Streamlit **Instrument list** tab shows one canonical row per tag with all drawing occurrences and downloadable JSON/CSV.

## Regex tag identification

A pattern accepting 1–5 letters, an optional separator, a 3–6 digit identifier and optional suffixes (`P-101`, `P4711`, `SV 104.01`) scans OCR text for tag-shaped substrings and classifies prefixes with an editable rule table: `P4711` → pump, `H1007` → heat exchanger, `T4750` → tank, `PV4712.02` → instrument/pressure valve. Pipe-service prefixes, nozzle labels and size labels (`N1`, `DN 800`, `P3.1415`) are excluded by boundary checks, standalone instrument labels like `PI` are exported as incomplete rather than invented, and unrecognized prefixes stay `unknown`. Classification is explicitly provisional — OCR confidence is not classification confidence.

## RAG backend

The FastAPI backend ingests `DATA/true_data` and `DATA/noisy_data`, chunks documents to JSON under `processed_data/`, embeds each chunk with **FastEmbed** (`BAAI/bge-small-en-v1.5`) and indexes vectors plus source metadata in **Qdrant** — local mode persisted under `outputs/qdrant/` by default, or a remote cluster via `QDRANT_CLUSTER_ENDPOINT` / `QDRANT_API_KEY`.

```bash
uv run uvicorn backend.main:app --reload   # API → http://127.0.0.1:8000/docs
uv run python -m app.ingestion.processor DATA/true_data
```

Routes: `POST /ingest`, `POST /ingest/file`, `GET /documents`, `GET /health`, `POST /search` (semantic retrieval, optional `source_type` filter) and `POST /query` (matching excerpts with filenames and chunk numbers). Text, Markdown, HTML, CSV, JSON, SVG, PDF, raster images, DOCX and PPTX are supported — SVG and raster P&IDs go through the same tiled RapidOCR pipeline, scanned PDFs through OCR per page.

## Agentic RAG chat workspace

The Streamlit app's sidebar now selects a workspace: **Agentic RAG chat** talks to the FastAPI agent (agent steps and retrieved context are displayed, with sample industrial questions and a fresh-thread button), or **P&ID OCR** for the drawing tools. The sample corpus is synthetic training data, not approved plant procedures — stated plainly in the app.

## SVG text versus OCR

Because the reference drawing carries embedded `<text>`, the notebook independently extracts all 245 text elements with a safe XML parser and compares them one-to-one against the OCR inventory: an earlier single-orientation baseline found **246 OCR regions with 172 exact normalized matches (70.2% occurrence coverage)**. The comparison is deliberately framed as an inventory check, not a spatial evaluation or character-accuracy score — and SVG extraction never repairs OCR results.

## Verification

A test suite guards the geometry and behavior rather than trusting eyeballs:

```bash
uv run python tests/check_ocr_geometry.py           # rotation mapping, tile edges, selection
uv run python tests/check_region_classification.py  # border/title/table zones, scale independence
uv run python tests/check_exact_search.py           # match boundaries, crops, fallback windows
uv run python tests/check_partial_search.py         # PI4712 → PI4712.01 regression
uv run python tests/check_instrument_grouping.py    # split tags, suffixes, ambiguity
uv run python tests/check_streamlit_flow.py         # real reference OCR through the UI
```

## Conclusion

The honest summary is in the README itself: these are observed OCR results, not measured recall against a labeled reference, and geometric proximity is not proof of equipment ownership. Tiny labels, rotated text and symbol intersections still cause misses. But the pipeline — tiled multi-orientation OCR, explicit region geometry, auditable overlap suppression, grouped instrument tags and evidence-producing search — is a solid foundation for the next milestone: answering engineering questions over this evidence with a real LLM in the loop.

[Source code](https://github.com/achafi/industrial_rag)
