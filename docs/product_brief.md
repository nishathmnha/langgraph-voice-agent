# Product Brief

## Goal

Build a minimal AI-assisted TODO manager that accepts text or voice commands and manages user tasks through a LangGraph backend.

## Stack

| Layer | Technology |
| --- | --- |
| Agent orchestration | LangGraph |
| API | FastAPI |
| Runtime | Python |
| Database | PostgreSQL |
| Input | Text and browser microphone |
| Speech-to-text | Provider-backed transcription service |

## MVP Scope

- Register a user with `username` and `api_key`.
- Generate a unique `tenant_id` from a username hash plus a random suffix.
- Capture text commands directly.
- Capture voice, convert speech to text, and process the transcript.
- Create, update, complete, delete, prioritize, and query TODO tasks.
- Store all task data with tenant isolation.
- Return short conversational responses to the UI.

## Out of Scope for MVP

- Wake-word activation.
- Continuous listening.
- Real-time voice conversation.
- Reminder notifications.
- Analytics dashboards.
- Text-to-speech responses.

## Example Commands

- Add gym at 7 PM.
- Mark backend task as completed.
- Remove yesterday's pending task.
- Update meeting time to 4 PM.
- What did I complete today?
