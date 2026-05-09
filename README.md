# Voice Todo Agent Documentation Pack

This repository contains the minimal planning documentation for a LangGraph-based voice-input todo system.

The system lets a user speak todos, update priorities, mark tasks completed, list todos, and store completion time logs.

There is no audio response in the current scope. The system returns transcripts, updated todos, time-log data, and short text responses for the UI.

## Contents

- `docs/minimal_langgraph_mvp.md` - The single source of truth for the minimal LangGraph MVP
- `diagrams/*.mmd` - Mermaid UML/architecture sources
- `exports/png/*.png` - Rendered diagram images
- `exports/pdf/minimal_langgraph_mvp.pdf` - The minimal PDF document
- `scripts/export_docs.py` - Simple offline Markdown-to-PDF exporter

## Main MVP Flow

1. User speaks through a push-to-talk UI.
2. Audio is transcribed to text.
3. LangGraph classifies intent.
4. The graph routes to add, update, complete, list, or clarify.
5. Todo tools update the database.
6. The UI receives updated todo state and a text response.
