from __future__ import annotations

import getpass
from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from voice_agent_todo.config import get_settings
from voice_agent_todo.graph import build_todo_graph
from voice_agent_todo.models import BaseState


class TodoAssistant:
    def __init__(self, thread_id: str | None = None) -> None:
        settings = get_settings()
        self.llm = ChatOpenAI(model=settings.model, api_key=settings.openai_api_key)
        self.memory = InMemorySaver()
        self.graph = build_todo_graph(self.llm, checkpointer=self.memory)

        logged_in_user = getpass.getuser()
        self.thread_id = thread_id or f"todo-assistant-{logged_in_user}"
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def invoke(self, messages: Iterable[BaseMessage] | str, todo_list=None) -> BaseState:
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        payload: BaseState = {"messages": list(messages)}
        if todo_list is not None:
            payload["todo_list"] = todo_list

        return self.graph.invoke(payload, config=self.config)

    def snapshot(self) -> BaseState:
        return self.graph.get_state(self.config).values
