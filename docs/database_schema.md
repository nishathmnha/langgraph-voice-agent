# Database Schema

## Multi-Tenant Rule

Every table contains `tenant_id`. Application code must filter by `tenant_id` for every read, update, and delete.

## users

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid | Primary key |
| tenant_id | text | Unique tenant key |
| username | text | User-provided name |
| api_key | text | Encrypted or hashed at rest |
| created_at | timestamptz | Creation timestamp |

## tasks

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid | Primary key |
| tenant_id | text | Required tenant scope |
| task_title | text | Short task title |
| task_description | text | Optional details |
| status | text | pending, completed, deleted |
| priority | text | low, medium, high |
| created_at | timestamptz | Creation timestamp |
| updated_at | timestamptz | Last update timestamp |
| completed_at | timestamptz | Completion timestamp |

## task_logs

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid | Primary key |
| tenant_id | text | Required tenant scope |
| task_id | uuid | Foreign key to tasks.id |
| action | text | create, update, complete, delete, query |
| timestamp | timestamptz | Log timestamp |

## Indexes

```sql
create unique index users_tenant_id_idx on users (tenant_id);
create index tasks_tenant_status_idx on tasks (tenant_id, status);
create index tasks_tenant_created_idx on tasks (tenant_id, created_at desc);
create index task_logs_tenant_task_idx on task_logs (tenant_id, task_id, timestamp desc);
```
