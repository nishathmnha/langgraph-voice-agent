# API Contract

## POST /users/register

Creates a user and returns a tenant ID.

```json
{
  "username": "nisha",
  "api_key": "provider-key"
}
```

```json
{
  "user_id": "uuid",
  "tenant_id": "tenant_9f4a12_4821",
  "username": "nisha"
}
```

## POST /agent/message

Processes typed input.

```json
{
  "tenant_id": "tenant_9f4a12_4821",
  "message": "Add gym at 7 PM"
}
```

```json
{
  "intent": "create_task",
  "response": "Added gym at 7 PM.",
  "tasks": []
}
```

## POST /agent/voice

Processes uploaded audio.

Form fields:

| Field | Type |
| --- | --- |
| tenant_id | text |
| audio | file |

Response:

```json
{
  "transcript": "Add gym at 7 PM",
  "intent": "create_task",
  "response": "Added gym at 7 PM.",
  "tasks": []
}
```

## GET /tasks

Returns tenant-scoped tasks.

Query parameters:

| Parameter | Example |
| --- | --- |
| tenant_id | tenant_9f4a12_4821 |
| status | pending |
| priority | high |

## Error Shape

```json
{
  "error": "tenant_id_required",
  "message": "A tenant_id is required for this operation."
}
```
