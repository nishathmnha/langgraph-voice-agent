# Minimal LangGraph TODO MVP

## Goal

Build a small LangGraph workflow that processes typed text or transcribed voice and updates a tenant-scoped TODO list.

```text
Voice or text input -> FastAPI -> optional STT -> LangGraph -> PostgreSQL -> UI response
```

## Minimal Screens

| Screen | Fields and controls |
| --- | --- |
| Registration/setup | API key, username |
| Main UI | Text input, microphone button, task list, response area |

## Tenant Setup

The backend generates `tenant_id` during registration:

```text
tenant_id = "tenant_" + sha256(username)[0:8] + "_" + random_4_digits
```

Every database table includes `tenant_id`.

## Minimal LangGraph Nodes

```text
START
  |
normalize_input
  |
detect_intent
  |
extract_entities
  |
match_task when needed
  |
decide_action
  |
execute_tool
  |
generate_response
  |
END
```

## Supported Actions

| Intent | Example |
| --- | --- |
| create_task | Add gym at 7 PM |
| update_task | Update meeting time to 4 PM |
| complete_task | Mark backend task as completed |
| delete_task | Remove yesterday's pending task |
| prioritize_task | Make backend task high priority |
| query_tasks | What did I complete today? |

## Keep It Simple

- Use one LangGraph workflow with task tools.
- Keep speech-to-text outside the graph.
- Scope every tool call by `tenant_id`.
- Log every create, update, complete, delete, and query action.
- Ask a clarification question only when task matching is uncertain.
