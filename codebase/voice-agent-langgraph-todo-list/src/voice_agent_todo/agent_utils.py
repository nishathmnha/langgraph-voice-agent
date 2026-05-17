from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage


def get_user_text(state: dict[str, Any]) -> str:
    message_list = state.get("messages", [])
    if message_list:
        return message_list[-1].content

    return state.get("user_input") or ""


def agent_json(agent: Any, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        content = messages[-1].content if messages else ""

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        content = str(content).strip()
        if content.startswith("```"):
            content = content.strip("`").removeprefix("json").strip()

        if not content.startswith("{") and "{" in content and "}" in content:
            content = content[content.find("{") : content.rfind("}") + 1]

        return json.loads(content)
    except Exception:
        return fallback
