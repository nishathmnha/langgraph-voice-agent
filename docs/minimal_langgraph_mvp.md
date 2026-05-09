# Minimal LangGraph Todo MVP

## Goal

Build a small LangGraph workflow that processes transcribed audio and updates a todo list.

Keep STT outside LangGraph:

```text
Audio input -> Speech-to-text -> LangGraph -> Todo tools -> Database -> Text/UI response
```

This is not a multi-agent system. It is a single LangGraph agent/workflow with tools.

## Minimal LangGraph Nodes

```text
START
  ↓
classify_intent
  ↓
route_intent
  ↓
[add_todo | update_todo | complete_todo | list_todos | clarify]
  ↓
generate_response
  ↓
END
```

## Nodes Needed

| Node | Purpose |
| --- | --- |
| `classify_intent` | Decide what the user wants: add, update, complete, list, or clarify. |
| `add_todo` | Extract tasks from transcript and save them. |
| `update_todo` | Change priority, due date, title, or notes. |
| `complete_todo` | Match spoken completed task to existing todo and mark it completed. |
| `list_todos` | Fetch today's pending/completed todos. |
| `clarify` | Ask a short question when task or detail is unclear. |
| `generate_response` | Return simple text for UI. |

## Minimal Tools

| Tool | Purpose |
| --- | --- |
| `create_todo(title, due_date, priority)` | Add new todo. |
| `list_todos(date, status)` | Get todos. |
| `update_todo(todo_id, fields)` | Change priority/details/status. |
| `complete_todo(todo_id, note)` | Mark completed and create time log. |
| `find_matching_todo(text)` | Fuzzy match spoken task to saved todo. |

## Minimal Intents

```python
ADD_TODO
UPDATE_TODO
COMPLETE_TODO
LIST_TODOS
CLARIFY
```

## Simple Flows

### Add Todo

```text
"I need to fix loan API today"
-> classify_intent: ADD_TODO
-> add_todo
-> create_todo()
-> response: "Added: Fix loan API."
```

### Update Priority

```text
"Make loan API high priority"
-> classify_intent: UPDATE_TODO
-> update_todo
-> find_matching_todo()
-> update_todo()
-> response: "Updated priority to high."
```

### Complete Todo

```text
"I completed the loan API task"
-> classify_intent: COMPLETE_TODO
-> complete_todo
-> find_matching_todo()
-> complete_todo()
-> response: "Marked loan API as completed."
```

### List Todos

```text
"What is left today?"
-> classify_intent: LIST_TODOS
-> list_todos()
-> response: pending todos
```

## Keep It Simple

- Use one LangGraph workflow.
- Use tools for database actions.
- Use fuzzy matching only when updating or completing existing todos.
- Ask clarification only when task matching or required details are unclear.
- Do not add TTS/audio response for the MVP.

