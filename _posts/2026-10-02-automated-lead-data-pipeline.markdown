---
title : "Automated lead data pipeline (map-based business collector)"
date : 2026-10-02
tags : [data engineering, geospatial, Streamlit, Python]
header :
  image : ""
excerpt : "Data engineering, OpenStreetMap, Streamlit"
---
[Source code](https://github.com/achafi/automated_lead_data_pipeline)

# Automated lead data pipeline: from a drawn rectangle to a clean CSV
*The objective of this project is to build an end-to-end lead data pipeline that a non-technical user can operate: draw a rectangle on a map, pick a business category, collect the OpenStreetMap records inside that area, then clean, deduplicate, review and validate the results — and download a CSV. No API key, no server, no database: one Streamlit app with a real pipeline behind it.*

## Introduction

Lead lists are usually bought, scraped with brittle scripts, or exported from a CRM. This project takes the opposite route: OpenStreetMap is a free, openly licensed dataset, and the interesting work is not the fetch — it is everything that happens *after* the fetch, which is where most ad-hoc lead scripts stop.

The app is a small but complete pipeline: **collect → summarize → detect duplicates → review → clean → validate → export**, with every stage producing a separate DataFrame so the original collected records are never mutated.

## Workflow

1. Choose a category: **Dentists, Pharmacies, Restaurants, Cafés or Hotels** (each maps to a fixed OSM tag key/value pair).
2. Pan the OpenStreetMap view and drag a **rectangle** with the draw tool.
3. Set the maximum results (default 50, range 1–200) and click **Collect Data**. Moving the map alone never fires an API request.
4. Inspect clustered markers, popups, the search summary and the table; click **Download original CSV**.

Drawing coordinates arrive as GeoJSON longitude/latitude pairs and are converted to Overpass's south, west, north, east order. Reversed drawing direction is supported, while malformed, zero-size, out-of-range and date-line-crossing rectangles are rejected. A single synchronous POST goes to `https://overpass-api.de/api/interpreter`, selecting nodes, ways and relations with `out body center <max>` — no pagination, no retry, no provider fallback.

## Summary cards and data quality

After a search, five summary cards report the number collected, records with any contact information, websites present, complete addresses, and potential duplicate groups — with counts and percentages that reflect only the collected fields (missing website data does not prove a business has no website).

## Duplicate detection

This is the heart of the pipeline. By default, potential duplicates are found by:

- normalized **phone numbers or website domains** within **500 m**,
- or **names** with ≥ **90% RapidFuzz** similarity within **150 m**,
- all requiring valid coordinates, with distances computed by the **Haversine** formula.

Matching pairs are connected into groups, so A–B and B–C count as one group of three even if A and C never matched directly. All three thresholds are sliders in the sidebar (distance 25–2,000 m, name similarity 70–100%), and recalculating happens locally on existing results — no new API requests, and the source records stay untouched.

## Review, deduplication and automatic filling

The **Potential Duplicate Review** section shows only flagged records, one expander per group, with comparison tables (contact info, address, coordinates, OSM identity) and a pair-level table explaining each connection. Confidence labels are transparent heuristics rather than probabilities: phone match 95, website 90, qualifying name-only match 80, identical OSM type + ID 100.

- Every group defaults to **Keep all records** — nothing is ever removed automatically.
- **Keep the recommended record only** (recommendations count populated fields, ties broken by website → phone → earliest row) or pick a specific record.
- **Create deduplicated results** builds a *separate* DataFrame and reports original / reviewed / excluded / final counts.
- Choosing a retention also authorizes **filling missing fields** in the retained record from group members — but only when all populated donor values agree exactly, never overwriting an existing value, never copying coordinates or OSM IDs, with a downloadable fill report for provenance.

The **Download original CSV** always exports the unchanged records; the deduplicated CSV appears only after creation, with a `_deduplicated.csv` suffix.

## Cleaning and validation

Four optional cleaning steps (whitespace collapsing, consistent blanks, contact-field trimming, country aliases → ISO alpha-2 codes) are applied to a copy, followed by **deterministic validation** of every populated value: latitude/longitude ranges, email format, URL sanity, and a basic phone format (7–15 digits, optional `+`, separators, optional extension).

The validation report counts issues by field and type and lists flagged records — but flags without removing. Missing values are excluded from validation entirely; completeness stays in the summary cards where it belongs.

## Demo mode

**Use demo data** loads a predefined rectangle over central London with 20 fictional businesses per category (100 total, covering nodes, ways and relations), deliberately seeded with duplicate chains, contact fallbacks, whitespace problems, malformed emails, out-of-range coordinates and empty fields — a self-contained scenario for a portfolio walkthrough. Demo collection makes **no Overpass request**, demo downloads carry a `demo_` prefix, and OSM IDs and links are blank so nobody mistakes fiction for a real listing.

## Architecture

```text
app.py                    Streamlit UI, session state, 5-minute search cache
categories.py             Single category → OSM tag mapping
overpass_client.py        Bounded HTTP request and OSM response mapping
duplicate_detection.py    Normalization, distance, pair matching, connected groups
duplicate_review.py       Evidence, recommendations, review state, separate results
data_completion.py        Retention-authorized fills and provenance
data_cleaning.py          Optional formatting on a separate DataFrame
data_validation.py        Read-only populated-value format reports
data_summary.py           Field-presence counts and percentages
map_utils.py              Bounding boxes, area guard, map, popups, filenames
tests/                    Mocked HTTP, geometry and workflow tests
```

Guardrails around the public API: a 25 km² area guard (enforced again in the client), a 200-element output cap, 25 s server / 35 s HTTP timeouts, and a five-minute in-memory cache of successful searches (failures are never cached).

## Running it

```bash
uv python install 3.12
uv sync
uv run streamlit run app.py
uv run pytest          # all HTTP mocked — tests never call the live API
```

No credentials or `.env` are needed. Data © OpenStreetMap contributors (ODbL); attribution stays visible on the map and in the exports.

## Limitations

- OSM coverage and contact details vary by area; results are raw elements, not verified leads, and a business can appear more than once as different element types.
- Five categories and one Overpass endpoint only; no accounts, database, workers or automation platform — this is an MVP, not a lead-generation service.
- Duplicate confidence is a matching heuristic, not a record-quality score.

## Conclusion

The pipeline's design rule is the interesting part: **every transformation happens on a copy, and only explicit review decisions exclude anything**. Between that, the Haversine + RapidFuzz grouping, the fill-provenance report and the mocked test suite, this project is as much about data engineering discipline as it is about the map.

[Source code](https://github.com/achafi/automated_lead_data_pipeline)
