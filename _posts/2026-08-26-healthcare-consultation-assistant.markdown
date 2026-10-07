---
title : "Healthcare consultation assistant"
date : 2026-08-26
tags : [LLM, FastAPI, Next.js, authentication, GenAI]
header :
  image : ""
excerpt : "LLM, FastAPI, Next.js, Clerk"
---
[Source code](https://github.com/achafi/healthcare_consultation_assistant)

# Healthcare consultation assistant
*The objective of this project is to build a full-stack, authenticated LLM application for a realistic clinical workflow: a doctor pastes their notes from a patient visit, and the system streams back a structured summary — visit summary, next steps, and a draft email to the patient in plain language. The hard part was not the prompt, it was everything around it: token verification, CORS, streaming, and two separate applications that have to deploy together.*

## Introduction

A single "ask the model" script does not survive contact with a real product. This project was an exercise in building the surrounding machinery properly:

- a **Next.js frontend** with Clerk authentication and a form for the visit notes,
- a **FastAPI backend** that verifies the Clerk access token before doing anything,
- **streaming** responses, so the doctor reads the summary as it is generated,
- and an explicit deployment story (Vercel, CORS allow-list, environment hygiene).

The system prompt locks the model into exactly three sections — *Summary of visit for the doctor's records*, *Next steps for the doctor*, and *Draft of email to patient in patient-friendly language* — so the output lands in a predictable structure instead of free-form prose.

## Architecture

The repository contains two independent applications:

```text
healthcare_consultation_assistant/
├── backend/
│   ├── api/index.py     # FastAPI app: /api endpoint, Clerk guard, SSE streaming
│   ├── requirements.txt
│   └── vercel.json      # Vercel Python runtime deployment
└── frontend/
    ├── pages/           # Next.js Pages Router
    └── public/
```

**Backend** (`backend/`): a FastAPI API that streams an OpenAI response.
**Frontend** (`frontend/`): a Next.js Pages Router application with Clerk authentication.

## Security model

Three things are enforced in the backend rather than trusted from the browser:

1. **Token verification.** `CLERK_JWKS_URL` is required at startup — the app refuses to boot without it. Requests pass through `ClerkHTTPBearer`, which verifies the Clerk access token against the JWKS endpoint, and the decoded `sub` claim is available for tracking and auditing.
2. **CORS allow-list.** `CORS_ALLOWED_ORIGINS` is a comma-separated list of exact frontend origins (defaulting to `http://localhost:3000`). The authenticated application never uses `*`.
3. **Minimal surface.** Only `POST` and `OPTIONS` methods and the `Authorization` / `Content-Type` headers are allowed, and the endpoint takes a typed `Visit` model (`patient_name`, `date_of_visit`, `notes`).

On the frontend, the usual rule is enforced in the documentation and configuration: variables prefixed `NEXT_PUBLIC_` are embedded in the browser bundle, so no secret ever uses that prefix.

## Streaming

The `/api` endpoint builds the prompt from the doctor's notes, calls OpenAI with `stream=True`, and returns a `StreamingResponse` with `media_type: text/event-stream`. Chunks are emitted line by line as SSE events, so the frontend renders the summary progressively instead of waiting for the whole completion. The README is explicit that every request calls OpenAI and may incur usage charges — an honest reminder that streaming demos are not free.

## Running it locally

**Backend** (Python 3.13 + [uv](https://docs.astral.sh/uv/)):

```bash
cd backend
uv venv --python 3.13
uv pip install -r requirements.txt

# backend/.env — never committed
# OPENAI_API_KEY=...
# CLERK_JWKS_URL=https://your-clerk-domain/.well-known/jwks.json
# CORS_ALLOWED_ORIGINS=http://localhost:3000

set -a; source .env; set +a
.venv/bin/uvicorn api.index:app --reload --host 127.0.0.1 --port 8000
```

**Frontend** (Node.js 22 LTS):

```bash
nvm use 22
cd frontend
npm ci

# frontend/.env.local — Clerk publishable/secret keys + NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

Then open <http://localhost:3000> (product page at `/product`), the backend at <http://127.0.0.1:8000/docs>, and the streaming endpoint at `POST /api`.

## Deployment

Both apps deploy to Vercel:

- Frontend: `NEXT_PUBLIC_API_URL=https://your-backend.vercel.app`
- Backend: `CLERK_JWKS_URL=...` and `CORS_ALLOWED_ORIGINS=https://your-frontend.vercel.app`

Because `NEXT_PUBLIC_*` values are compiled into the browser bundle, changing one requires restarting `npm run dev` (or redeploying).

## Production checks

```bash
cd frontend && npm run lint && npm run build
# backend import check with .env loaded
.venv/bin/python -c "from api.index import app; print(app.title)"
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js (Pages Router), React |
| Auth | Clerk (sign-in, user button) |
| Backend | FastAPI, Pydantic, `fastapi-clerk-auth` |
| LLM | OpenAI `gpt-5-nano`, streamed as SSE |
| Streaming | `StreamingResponse` + browser SSE consumption |
| Tooling | uv (Python 3.13), npm, Node.js 22 LTS |
| Deployment | Vercel (Python runtime + Node runtime) |

## Conclusion

The model call in this project is about ten lines; everything else — verified tokens, an origin allow-list, two runtimes, streaming, environment hygiene — is what makes it an application. It is also a good reminder that healthcare-flavoured LLM features live or die on output structure, which is why the three-heading system prompt matters more than the model choice.

[Source code](https://github.com/achafi/healthcare_consultation_assistant)
