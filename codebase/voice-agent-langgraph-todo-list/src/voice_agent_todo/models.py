from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TaskDetails(TypedDict, total=False):
    task: str
    priority: int
    completed: bool
    active: bool


class TodoListExtraction(TypedDict):
    tasks: list[TaskDetails]


class ActionDecision(TypedDict):
    node_name: Literal["add_tasks_to_todo_list", "update_task", "update_status"]


def _normalize_task(task: TaskDetails, priority: int) -> TaskDetails:
    return {
        "task": task.get("task", "Untitled task"),
        "priority": priority,
        "completed": bool(task.get("completed", False)),
        "active": bool(task.get("active", True)),
    }


def replace_all_tasks(tasks: list[TaskDetails]) -> list[dict[str, Any]]:
    return [{"__replace_all__": True}, *tasks]


def merge_todo_list(
    existing: list[TaskDetails],
    new: list[TaskDetails] | list[dict[str, Any]],
) -> list[TaskDetails]:
    incoming = list(new or [])

    if incoming and incoming[0].get("__replace_all__"):
        merged = [task for task in incoming[1:] if task.get("task")]
    else:
        merged = list(existing or [])

        for new_task in incoming:
            new_task_name = new_task.get("task", "").lower().strip()
            match_index = None

            for index, existing_task in enumerate(merged):
                existing_task_name = existing_task.get("task", "").lower().strip()
                if new_task_name and new_task_name == existing_task_name:
                    match_index = index
                    break

            if match_index is None:
                merged.append(new_task)
            else:
                merged[match_index] = {**merged[match_index], **new_task}

    return [_normalize_task(task, index) for index, task in enumerate(merged, start=1)]


class BaseState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    action: Literal["add_tasks_to_todo_list", "update_task", "update_status"]
    todo_list: Annotated[list[TaskDetails], merge_todo_list]
    response: str
