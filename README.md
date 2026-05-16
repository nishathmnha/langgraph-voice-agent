# Voice Agent TODO List

Minimal documentation pack for an AI-assisted TODO management system using LangGraph, FastAPI, Python, and PostgreSQL.

The product supports text input and voice input. Voice is transcribed to text, then the same LangGraph workflow handles task creation, updates, completion, deletion, prioritization, and natural-language queries.

## Contents

- `docs/product_brief.md` - compact product scope and MVP boundaries
- `docs/architecture.md` - backend, frontend, voice, and data architecture
- `docs/database_schema.md` - PostgreSQL multi-tenant tables
- `docs/api_contract.md` - minimal FastAPI endpoints and payloads
- `docs/langgraph_workflow.md` - LangGraph nodes, routing, and tool responsibilities
- `docs/minimal_langgraph_mvp.md` - one-page implementation summary
- `diagrams/*.mmd` - Mermaid UML and architecture diagram sources
- `exports/png/*.png` - rendered diagram PNG files
- `exports/pdf/*.pdf` - rendered document and diagram PDF files
- `scripts/export_docs.py` - offline exporter for Markdown and Mermaid assets

## MVP Flow

1. User registers with `username` and `api_key`.
2. Backend generates a `tenant_id` from the username hash plus a random suffix.
3. User submits either text or voice.
4. Voice input is transcribed before entering the LangGraph workflow.
5. LangGraph detects intent and routes to the correct task tool.
6. Tools read and write PostgreSQL rows scoped by `tenant_id`.
7. FastAPI returns the response text and updated task state to the UI.
