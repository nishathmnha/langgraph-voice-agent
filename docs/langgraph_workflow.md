# LangGraph Workflow

## State

```python
class AgentState(TypedDict):
    tenant_id: str
    raw_input: str
    transcript: str | None
    intent: str | None
    entities: dict
    matched_task_id: str | None
    tool_result: dict | None
    response: str | None
```

## Nodes

| Node | Purpose |
| --- | --- |
| normalize_input | Clean text and resolve relative dates where possible. |
| detect_intent | Classify the request as create, update, complete, delete, prioritize, query, or clarify. |
| extract_entities | Extract title, time, status, priority, and date filters. |
| match_task | Find the best existing task for update, completion, or deletion. |
| decide_action | Route to the correct task tool. |
| execute_tool | Run tenant-scoped database operation. |
| generate_response | Produce a short UI response. |

## Tools

| Tool | Responsibility |
| --- | --- |
| create_task | Insert a task row and write a task log. |
| update_task | Update title, description, due details, status, or priority. |
| complete_task | Set status to completed and set completed_at. |
| delete_task | Soft-delete or remove a task. |
| query_tasks | Return tasks filtered by status, date, or priority. |
| log_task_action | Insert task_logs entry for audit/history. |

## Intent Values

```text
create_task
update_task
complete_task
delete_task
prioritize_task
query_tasks
clarify
```

## Clarification Rule

Ask a clarification question when the command references an existing task and no confident match is found.
