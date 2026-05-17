from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from voice_agent_todo.agent_utils import agent_json, get_user_text
from voice_agent_todo.models import (
    ActionDecision,
    BaseState,
    TaskDetails,
    TodoListExtraction,
    replace_all_tasks,
)


ADD_TASK_SIGNALS = (
    "add these tasks",
    "add a task",
    "add a new task",
    "add tasks",
    "add task",
    "create todo list",
    "create to-do list",
    "create task",
    "todo list",
    "to-do list",
    "today i need",
    "i need to",
    "my plan",
    "plan for today",
    "tasks:",
)
STATUS_SIGNALS = (
    "done",
    "complete",
    "completed",
    "finished",
    "pending",
    "not done",
    "reopen",
    "remove",
    "delete",
    "archive",
    "inactive",
    "restore",
    "reactivate",
)
EDIT_SIGNALS = (
    "change",
    "edit",
    "rename",
    "set priority",
    "priority",
    "prioritize",
    "swap",
    "reorder",
    "move",
    "raise",
    "lower",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "please",
    "task",
    "tasks",
    "the",
    "this",
    "to",
}
CARDINALS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _clean_voice_text(text: str) -> str:
    cleaned = re.sub(r"^\s*final\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*hi[,.\s]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*at\s+(\d+|three)\s+more\s+tasks?\b", r"add \1 more tasks", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_task_text(text: str) -> str:
    task = text.strip(" .,\n\t")
    task = re.sub(r"^(add|at|create)\s+(\d+|three|these|more|\s)*\s*tasks?\s*:?\s*", "", task, flags=re.IGNORECASE)
    task = re.sub(r"^(first|then|next|also|and)\s+", "", task, flags=re.IGNORECASE)
    task = re.sub(r"^i(?:'m| am)?\s+going\s+to\s+", "", task, flags=re.IGNORECASE)
    task = re.sub(r"^i\s+need\s+to\s+", "", task, flags=re.IGNORECASE)
    task = re.sub(r"\bblock\b", "blog", task, flags=re.IGNORECASE)
    task = re.sub(r"\bwhistle\s+ai\s+sdk\b", "Vercel AI SDK", task, flags=re.IGNORECASE)
    return task.strip(" .,\n\t")


def _stem_token(token: str) -> str:
    if token in {"updated", "updating", "updates"}:
        return "update"
    if token in {"completed", "completing"}:
        return "complete"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem_token(token) for token in raw_tokens if token not in STOPWORDS]


def _word_to_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    return CARDINALS.get(value) or ORDINALS.get(value)


def _extract_referenced_priorities(text: str) -> set[int]:
    lowered_text = text.lower()
    referenced_priorities: set[int] = set()

    for match in re.finditer(r"\b(?:priority|task|item|number|#)\s*(\d+)\b", lowered_text):
        referenced_priorities.add(int(match.group(1)))

    for pattern in (
        r"\b(?:priority|task|item)\s+number\s+([a-z0-9]+)\b",
        r"\bnumber\s+([a-z0-9]+)\s+(?:task|item|priority)\b",
        r"\b(?:priority|task|item)\s+([a-z0-9]+)\b",
    ):
        for match in re.finditer(pattern, lowered_text):
            number = _word_to_number(match.group(1))
            if number is not None:
                referenced_priorities.add(number)

    for ordinal, priority in ORDINALS.items():
        if re.search(rf"\b{ordinal}\b", lowered_text):
            referenced_priorities.add(priority)

    return referenced_priorities


def _task_match_score(user_text: str, task_text: str) -> float:
    query_tokens = set(_tokens(user_text))
    task_tokens = set(_tokens(task_text))
    if not query_tokens or not task_tokens:
        return 0.0

    overlap = len(query_tokens & task_tokens)
    coverage = overlap / len(task_tokens)
    precision = overlap / len(query_tokens)
    query_norm = " ".join(sorted(query_tokens))
    task_norm = " ".join(sorted(task_tokens))
    fuzzy = SequenceMatcher(None, query_norm, task_norm).ratio()
    return max(coverage, precision, fuzzy * 0.75)


def _find_named_task_indexes(user_text: str, todo_list: list[TaskDetails]) -> list[int]:
    lowered_text = user_text.lower()
    matches: list[int] = []
    for index, task in enumerate(todo_list):
        task_text = task.get("task", "").lower().strip()
        if not task_text:
            continue
        if task_text in lowered_text:
            matches.append(index)
    return matches


def _find_matching_task_indexes(user_text: str, todo_list: list[TaskDetails]) -> list[int]:
    lowered_text = user_text.lower()
    if any(word in lowered_text for word in ("all tasks", "every task", "everything")):
        return list(range(len(todo_list)))

    referenced_priorities = _extract_referenced_priorities(user_text)

    if referenced_priorities:
        priority_matches = [
            index
            for index, task in enumerate(todo_list)
            if int(task.get("priority", -1)) in referenced_priorities
        ]
        if priority_matches:
            return priority_matches

    named_matches = _find_named_task_indexes(user_text, todo_list)
    if named_matches:
        return named_matches

    scored = [
        (index, _task_match_score(user_text, task.get("task", "")))
        for index, task in enumerate(todo_list)
    ]
    scored = [(index, score) for index, score in scored if score >= 0.45]
    if not scored:
        return []

    best_score = max(score for _, score in scored)
    return [index for index, score in scored if score >= max(0.45, best_score - 0.12)]


def _looks_like_new_task_list(text: str) -> bool:
    lowered_text = _clean_voice_text(text).lower()
    return (
        any(signal in lowered_text for signal in ADD_TASK_SIGNALS)
        or bool(re.search(r"\b(add|at)\s+(\d+|one|two|three|four|five)\s+more\s+tasks?\b", lowered_text))
    )


def _starts_with_any(text: str, phrases: tuple[str, ...]) -> bool:
    cleaned = _clean_voice_text(text).lower().strip()
    return any(cleaned.startswith(phrase) for phrase in phrases)


def _is_explicit_status_command(text: str) -> bool:
    return _starts_with_any(
        text,
        (
            "remove",
            "delete",
            "archive",
            "mark",
            "complete",
            "finish",
            "set",
            "restore",
            "reactivate",
            "reopen",
        ),
    )


def _is_explicit_edit_command(text: str) -> bool:
    return _starts_with_any(
        text,
        (
            "swap",
            "reorder",
            "change",
            "edit",
            "rename",
            "move",
            "raise",
            "lower",
            "prioritize",
            "set priority",
            "update task",
            "update the task",
        ),
    )


def _status_kind(text: str) -> str | None:
    changes = _extract_status_changes(text)
    if "active" in changes:
        return "activate" if changes["active"] else "deactivate"
    if "completed" in changes:
        return "complete" if changes["completed"] else "incomplete"
    return None


def _extract_status_changes(text: str) -> dict[str, bool]:
    lowered_text = text.lower()
    changes: dict[str, bool] = {}

    if (
        any(phrase in lowered_text for phrase in ("not active", "inactive", "deactivate", "disabled"))
        or re.search(r"\b(?:active|activation)\b.*\b(?:no|false|off)\b", lowered_text)
        or re.search(r"\b(?:set|make)\b.*\b(?:active|activation)\b.*\b(?:no|false|off)\b", lowered_text)
    ):
        changes["active"] = False
    elif (
        any(phrase in lowered_text for phrase in ("restore", "reactivate", "bring back"))
        or re.search(r"\b(?:active|activation)\b.*\b(?:yes|true|on)\b", lowered_text)
        or re.search(r"\b(?:set|make)\b.*\b(?:active|activation)\b.*\b(?:yes|true|on)\b", lowered_text)
    ):
        changes["active"] = True
    elif any(phrase in lowered_text for phrase in ("remove", "delete", "archive", "no longer needed")):
        changes["active"] = False

    if (
        any(phrase in lowered_text for phrase in ("not done", "pending", "incomplete", "reopen", "undo"))
        or re.search(r"\b(?:complete|completed|completion)\b.*\b(?:no|false)\b", lowered_text)
        or re.search(r"\b(?:status of completion|completion status)\b.*\b(?:no|false)\b", lowered_text)
    ):
        changes["completed"] = False
    elif (
        any(phrase in lowered_text for phrase in ("done", "completed", "finished"))
        or re.search(r"\bmark\b.*\bcomplete(?:d)?\b", lowered_text)
        or re.search(r"\b(?:complete|completed|completion)\b.*\b(?:yes|true)\b", lowered_text)
        or re.search(r"\b(?:status of completion|completion status)\b.*\b(?:yes|true)\b", lowered_text)
        or re.search(r"\bcomplete\b", lowered_text)
    ):
        changes["completed"] = True

    return changes


def _split_status_clauses(text: str) -> list[str]:
    normalized = _clean_voice_text(text)
    normalized = re.sub(r"\b(?:and then|then move on to|move on to|after that|next)\b", "|", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[.;]", "|", normalized)
    normalized = re.sub(r",\s*", "|", normalized)
    clauses = [clause.strip(" \n\t") for clause in normalized.split("|")]
    return [clause for clause in clauses if clause]


def _extract_status_operations(user_text: str, todo_list: list[TaskDetails]) -> list[tuple[list[int], dict[str, bool]]]:
    operations: list[tuple[list[int], dict[str, bool]]] = []
    current_targets: list[int] = []

    for clause in _split_status_clauses(user_text):
        clause_targets = _find_matching_task_indexes(clause, todo_list)
        if clause_targets:
            current_targets = clause_targets

        changes = _extract_status_changes(clause)
        if not changes:
            continue

        targets = current_targets or clause_targets
        if not targets:
            continue

        operations.append((targets, changes))

    return operations


def _looks_like_task_edit(text: str) -> bool:
    lowered_text = text.lower()
    return any(signal in lowered_text for signal in EDIT_SIGNALS)


def _simple_task_extraction(user_text: str) -> list[TaskDetails]:
    text = _clean_voice_text(user_text)

    split_markers = (
        "add these tasks:",
        "add tasks:",
        "add task:",
        "add three more tasks",
        "add 3 more tasks",
        "tasks:",
        "today i need to",
        "today i need",
        "i need to",
        "my plan for today is",
        "my plan is",
        "plan for today is",
    )

    lowered_text = text.lower()
    for marker in split_markers:
        marker_index = lowered_text.find(marker)
        if marker_index >= 0:
            text = text[marker_index + len(marker) :]
            break

    text = re.sub(r"\bthen\b", ",", text, flags=re.IGNORECASE)
    text = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)
    candidates = [_clean_task_text(part) for part in text.split(",")]
    tasks = [candidate for candidate in candidates if len(candidate.split()) >= 2]

    return [
        {"task": task, "priority": index, "completed": False, "active": True}
        for index, task in enumerate(tasks, start=1)
    ]


def _renumber_tasks(tasks: list[TaskDetails]) -> list[TaskDetails]:
    return [
        {**task, "priority": index}
        for index, task in enumerate(
            sorted(tasks, key=lambda task: task.get("priority", 50)),
            start=1,
        )
    ]


def _swap_or_reverse_priorities(user_text: str, todo_list: list[TaskDetails]) -> list[TaskDetails] | None:
    lowered_text = user_text.lower()
    if "swap" not in lowered_text and "reverse" not in lowered_text:
        return None
    if "priority" not in lowered_text and "priorities" not in lowered_text:
        return None

    active_tasks = [task for task in todo_list if task.get("active", True)]
    if len(active_tasks) < 2:
        return todo_list

    reversed_priorities = [
        task.get("priority", index)
        for index, task in enumerate(sorted(active_tasks, key=lambda task: task.get("priority", 50)), start=1)
    ][::-1]

    updated: list[TaskDetails] = []
    active_index = 0
    for task in todo_list:
        if task.get("active", True):
            updated.append({**task, "priority": reversed_priorities[active_index]})
            active_index += 1
        else:
            updated.append(task)

    return _renumber_tasks(updated)


def build_todo_graph(llm: Any, checkpointer: Any | None = None):
    agent = create_agent(model=llm)

    def add_tasks_to_todo_list(state: BaseState) -> Command:
        user_text = get_user_text(state)
        current_todo_list = state.get("todo_list", [])
        current_max_priority = max(
            (task.get("priority", 0) for task in current_todo_list),
            default=0,
        )

        prompt = f"""
Extract a minimal todo list from the user's message.
Return only valid JSON in this exact shape:
{{"tasks": [{{"task": "task text", "priority": 1, "completed": false, "active": true}}]}}

Rules:
- The user message may have more than one task.
- Do not hallucinate subtasks if input defines the tasks clearly.
- Prioritize these tasks based on the user's order and importance.
- Return only concrete tasks the user plans to do.
- priority must be an integer from 1 to 50.
- Use 1 for the most important or earliest task in this batch.
- completed must always be false.
- active must always be true.
- Correct obvious speech-to-text mistakes in developer terms.
- Examples: "block" may mean "blog"; "whistle AI SDK" may mean "Vercel AI SDK"; "at 3 more task" means "add 3 more tasks".
- If the user says they are adding 3 tasks but only clearly gives 2, return only the clear tasks.
- Do not keep filler like "at 3 more task" in the task text.

User message:
{user_text}
"""

        result = agent_json(agent, prompt, {"tasks": []})
        todo_list: list[TaskDetails] = result.get("tasks", [])

        if not todo_list and _looks_like_new_task_list(user_text):
            todo_list = _simple_task_extraction(user_text)
            print(f"[todo-graph] deterministic task extraction: {todo_list}", flush=True)

        if not todo_list and user_text:
            todo_list = [{"task": user_text, "priority": 1, "completed": False, "active": True}]

        todo_list = sorted(todo_list, key=lambda task: task.get("priority", 50))
        for index, task in enumerate(todo_list, start=1):
            task["task"] = _clean_task_text(task.get("task", ""))
            task["priority"] = current_max_priority + index
            task["completed"] = bool(task.get("completed", False))
            task["active"] = True

        return Command(
            update={
                "todo_list": todo_list,
                "response": "Initial todo list created.",
            }
        )

    def update_task(state: BaseState) -> Command:
        user_text = get_user_text(state)
        todo_list = state.get("todo_list", [])

        reordered_todo_list = _swap_or_reverse_priorities(user_text, todo_list)
        if reordered_todo_list is not None:
            return Command(
                update={
                    "todo_list": replace_all_tasks(reordered_todo_list),
                    "response": "Task priorities updated.",
                }
            )

        prompt = f"""
Find the proper task or tasks to update, then return the full updated todo list.
Return only valid JSON in this exact shape:
{{"tasks": [{{"task": "task text", "priority": 1, "completed": false, "active": true}}]}}

Rules:
- Keep tasks that are not mentioned unchanged.
- Update task text or priority only when the user asks for it.
- Do not invent new tasks.
- priority must be an integer from 1 to 50.
- completed must stay true or false.
- active must stay true or false.

Current todo list:
{todo_list}

User message:
{user_text}
"""

        result = agent_json(agent, prompt, {"tasks": todo_list})
        updated_todo_list = result.get("tasks", todo_list)

        return Command(
            update={
                "todo_list": replace_all_tasks(updated_todo_list),
                "response": "Task updated.",
            }
        )

    def update_status(state: BaseState) -> Command:
        user_text = get_user_text(state)
        todo_list = state.get("todo_list", [])
        operations = _extract_status_operations(user_text, todo_list)

        if operations:
            updated_todo_list = [dict(task) for task in todo_list]
            for matching_indexes, changes in operations:
                for index in matching_indexes:
                    if index < 0 or index >= len(updated_todo_list):
                        continue
                    if "active" in changes:
                        updated_todo_list[index]["active"] = changes["active"]
                    if "completed" in changes:
                        updated_todo_list[index]["completed"] = changes["completed"]

            return Command(
                update={
                    "todo_list": replace_all_tasks(updated_todo_list),
                    "response": "Task status updated.",
                }
            )

        prompt = f"""
Find the proper task or tasks to update completed/active status, then return the full updated todo list.
Return only valid JSON in this exact shape:
{{"tasks": [{{"task": "task text", "priority": 1, "completed": true, "active": true}}]}}

Rules:
- Only change completed or active status.
- completed true means done, completed, finished, or complete.
- completed false means not done, pending, incomplete, reopen, or undo.
- active false means remove, delete, archive, inactive, or no longer needed.
- active true means restore, reactivate, active, or bring back.
- Keep task text and priority unchanged.
- Do not invent new tasks.

Current todo list:
{todo_list}

User message:
{user_text}
"""

        result = agent_json(agent, prompt, {"tasks": todo_list})
        updated_todo_list = result.get("tasks", todo_list)

        if _status_kind(user_text) and (not updated_todo_list or updated_todo_list == todo_list):
            return Command(
                update={
                    "todo_list": replace_all_tasks(todo_list),
                    "response": "I could not find a matching task to update.",
                }
            )

        return Command(
            update={
                "todo_list": replace_all_tasks(updated_todo_list),
                "response": "Task status updated.",
            }
        )

    def decide_action(state: BaseState) -> Command:
        user_text = get_user_text(state)
        lowered_text = user_text.lower()

        is_new_task_list = _looks_like_new_task_list(user_text)
        is_status_update = _status_kind(user_text) is not None or any(signal in lowered_text for signal in STATUS_SIGNALS)
        is_task_edit = _looks_like_task_edit(user_text)

        if is_status_update and (_is_explicit_status_command(user_text) or not is_new_task_list):
            print("[todo-graph] route=update_status reason=status-command", flush=True)
            return Command(update={"action": "update_status"}, goto="update_status")

        if is_task_edit and (_is_explicit_edit_command(user_text) or not is_new_task_list):
            print("[todo-graph] route=update_task reason=edit-command", flush=True)
            return Command(update={"action": "update_task"}, goto="update_task")

        if is_new_task_list:
            print("[todo-graph] route=add_tasks_to_todo_list reason=new-task-list", flush=True)
            return Command(
                update={"action": "add_tasks_to_todo_list"},
                goto="add_tasks_to_todo_list",
            )

        prompt = f"""
Get the user input and decide which node to route to.
Return only valid JSON in this exact shape:
{{"node_name": "add_tasks_to_todo_list"}}

Current node list:
add_tasks_to_todo_list, update_task, update_status

Current node details:
add_tasks_to_todo_list: use when the user gives a new task list, describes today's plan, or asks to create/add tasks.
update_task: use when the user wants to edit task text, change priority, swap priorities, reorder, move, raise, or lower a task.
update_status: use when the user says a task is done, pending, removed, deleted, archived, restored, or active.

Important:
- If the user is listing tasks they plan to do, choose add_tasks_to_todo_list even if one task contains words like update, write, test, or clean.
- If the user says swap/reorder/change priority, choose update_task.
- If the user says remove/delete/archive/complete/done/pending, choose update_status.
- Choose update_task only when the user is modifying an existing task.

User input:
{user_text}
"""

        result: ActionDecision = agent_json(agent, prompt, {})  # type: ignore[assignment]
        node_name = result.get("node_name")

        if not node_name:
            if is_task_edit:
                node_name = "update_task"
            else:
                node_name = "add_tasks_to_todo_list"

        if node_name not in {"add_tasks_to_todo_list", "update_task", "update_status"}:
            node_name = "add_tasks_to_todo_list"

        print(f"[todo-graph] route={node_name} reason=agent-or-fallback", flush=True)
        return Command(update={"action": node_name}, goto=node_name)

    graph = StateGraph(BaseState)
    graph.add_node("decide_action", decide_action)
    graph.add_node("add_tasks_to_todo_list", add_tasks_to_todo_list)
    graph.add_node("update_task", update_task)
    graph.add_node("update_status", update_status)

    graph.add_edge(START, "decide_action")
    graph.add_edge("add_tasks_to_todo_list", END)
    graph.add_edge("update_task", END)
    graph.add_edge("update_status", END)

    return graph.compile(checkpointer=checkpointer)


def make_message(text: str) -> HumanMessage:
    return HumanMessage(content=text)
