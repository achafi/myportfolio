---
title : "IdeaGen Pro — AI SaaS with subscription gating"
date : 2026-08-24
categories : [agentic-ai]
tags : [SaaS, Next.js, TypeScript, FastAPI, GenAI]
header :
  image : ""
excerpt : "SaaS, Next.js, Clerk, streaming AI"
---
[Source code](https://github.com/achafi/ai_saas_demo)

# IdeaGen Pro — an AI SaaS product with subscription-gated streaming
*The objective of this project is to build the full skeleton of a modern AI SaaS product: a marketing landing page, authentication, a paid subscription plan that actually gates the product, and an AI feature whose output streams into the browser as server-sent events and renders as Markdown. It is a small app, but it touches every layer a real AI SaaS needs.*

## Introduction

"AI SaaS" projects often turn out to be a chat box with a paywall drawn in CSS. This one was built to practice the parts that are easy to skip and expensive to retrofit later: real authentication, a plan check that runs on the client *and* on the protected route, a streaming API instead of a spinner that blocks for thirty seconds, and a landing page that leads somewhere.

The product is **IdeaGen Pro**: you sign in, the premium plan unlocks the generator, and the backend streams a structured business idea for the AI-agent economy, token by token.

## Features

- **Landing page** — hero section, gradient typography, sign-in / sign-up with Clerk in modal mode, and a pricing preview for the $10/month premium plan.
- **Authentication** — Clerk `SignInButton`, `SignedIn` / `SignedOut` regions and `UserButton`, so anonymous visitors and signed-in users see different navigation.
- **Subscription gating** — the product page wraps the generator in Clerk's `<Protect plan="premium_subscription">`. Without the plan, users get Clerk's hosted `<PricingTable />` instead of the feature.
- **Streaming AI output** — the browser opens a server-sent events connection with a Clerk JWT in the `Authorization` header and appends chunks to a buffer as they arrive.
- **Markdown rendering** — the streamed buffer is rendered live with `react-markdown`, `remark-gfm` and `remark-breaks`, so headings and bullet points appear as they are generated.

## How it works

1. A visitor lands on `/` and signs in with the Clerk modal. Signed-in users get an "Go to App" button and their avatar.
2. `/product` renders `<Protect plan="premium_subscription">`. If the user has no subscription, the fallback UI shows the Clerk pricing table; the generator is never mounted.
3. Once protected, the `IdeaGenerator` component fetches a Clerk JWT with `getToken()` and calls `fetchEventSource('/api', { headers: { Authorization: Bearer <jwt> } })`.
4. The `/api` endpoint is a **FastAPI** function running on the Vercel Python runtime. It creates an OpenAI client, sends a prompt to `gpt-5-nano` with `stream=True`, and returns a `StreamingResponse` with `media_type: text/event-stream`.
5. Each SSE message carries one line of the completion; the client appends `ev.data` to a buffer and re-renders the Markdown on every message.

The result is a plain HTTP connection doing the work of a WebSocket: no polling, no fixed "generating..." delay, and the user reads the idea while it is still being written.

## Project structure

```text
pages/
├── index.tsx          # Landing page: hero, pricing preview, Clerk sign-in
├── product.tsx        # Protected generator + PricingTable fallback
├── _app.tsx           # Clerk provider wiring
└── _document.tsx
api/
└── index.py           # FastAPI SSE endpoint — OpenAI streaming (Vercel Python runtime)
public/                # Static assets
styles/                # Tailwind CSS
package.json           # Next.js 16, React 19, Clerk, react-markdown
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js (Pages Router), React 19, TypeScript |
| Styling | Tailwind CSS 4 + typography plugin |
| Auth & billing | Clerk (sign-in, user button, `<Protect>`, `<PricingTable>`) |
| Streaming | `@microsoft/fetch-event-source` (SSE client) |
| Markdown | `react-markdown` + `remark-gfm` + `remark-breaks` |
| Backend | FastAPI on the Vercel Python runtime |
| LLM | OpenAI `gpt-5-nano`, streamed token by token |

## Running locally

```bash
npm ci
npm run dev
```

Open <http://localhost:3000>. You need a Clerk application (publishable + secret keys) and an OpenAI key configured for the deployment; the Python `/api` function runs on Vercel's Python runtime, so the fastest end-to-end path is deploying the project to Vercel with both sets of keys set as environment variables. The repo includes the Vercel cache from real deployments.

## Limitations and what I would add next

- The idea generator runs once on mount and is not a conversation — there is no chat history or follow-up context.
- The subscription is checked by Clerk client-side guards; a production deployment should also verify the plan server-side before spending tokens on the OpenAI call.
- One hardcoded prompt, one model, no rate limiting and no usage metering.

Those are the natural next milestones: server-side plan verification, a usage quota, and turning the single-shot generator into a real conversation with streamed, persisted turns.

[Source code](https://github.com/achafi/ai_saas_demo)
