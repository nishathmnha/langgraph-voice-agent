# Architecture

## Overview

The frontend provides registration, text input, and microphone capture. FastAPI owns authentication context, transcription requests, LangGraph execution, and database access. PostgreSQL stores users, tasks, and task logs with `tenant_id` on every table.

```text
Frontend -> FastAPI -> Speech-to-text -> LangGraph -> Task tools -> PostgreSQL
```

## Frontend

- Screen 1: registration/setup with `api_key` and `username`.
- Screen 2: main task UI with text input and microphone button.
- Audio is recorded in the browser and uploaded to the backend.
- The frontend renders the transcript, agent response, and updated task list.

## Backend

- `POST /users/register` creates the user and tenant.
- `POST /agent/message` processes typed text.
- `POST /agent/voice` uploads audio, transcribes it, and processes the transcript.
- `GET /tasks` returns tasks for the active tenant.
- All task operations are performed by LangGraph tools.

## Tenant Isolation

Every database query must include `tenant_id`. The API resolves the tenant from the registered user/session and passes it into LangGraph state. Tools may not operate without a tenant ID.

## Suggested Project Structure

```text
taskgraph-ai/
  backend/
    agents/
    graph/
    api/
    services/
    database/
    models/
    main.py
  frontend/
  docker/
  docs/
  diagrams/
  exports/
  requirements.txt
  docker-compose.yml
```
