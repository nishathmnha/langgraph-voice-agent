from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from voice_agent_todo.graph import build_todo_graph


class FakeAgent:
    def invoke(self, payload):
        prompt = payload["messages"][-1].content
        if "node_name" in prompt:
            user_text = prompt.split("User input:", 1)[-1].lower()
            if "done" in user_text or "remove" in user_text:
                content = '{"node_name": "update_status"}'
            elif "priority" in user_text:
                content = '{"node_name": "update_task"}'
            else:
                content = '{"node_name": "add_tasks_to_todo_list"}'
        elif "active false" in prompt:
            content = '{"tasks": [{"task": "buy milk", "priority": 1, "completed": false, "active": false}]}'
        elif "Find the proper task" in prompt:
            content = '{"tasks": [{"task": "buy milk", "priority": 1, "completed": false, "active": true}]}'
        else:
            content = '{"tasks": [{"task": "buy milk", "priority": 1, "completed": false, "active": true}]}'

        return {"messages": [HumanMessage(content=content)]}


class FakeLlm:
    pass


def test_graph_add_update_and_remove(monkeypatch):
    import voice_agent_todo.graph as graph_module

    monkeypatch.setattr(graph_module, "create_agent", lambda model: FakeAgent())
    graph = build_todo_graph(FakeLlm(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test"}}

    state = graph.invoke(
        {"messages": [HumanMessage(content="add buy milk")], "todo_list": []},
        config=config,
    )
    assert state["todo_list"][0]["task"] == "buy milk"
    assert state["todo_list"][0]["active"] is True

    state = graph.invoke(
        {"messages": [HumanMessage(content="update buy milk priority 1")]},
        config=config,
    )
    assert state["action"] == "update_task"

    state = graph.invoke(
        {"messages": [HumanMessage(content="remove buy milk")]},
        config=config,
    )
    assert state["action"] == "update_status"
    assert state["todo_list"][0]["active"] is False


def test_today_plan_routes_to_add_even_when_task_contains_update(monkeypatch):
    import voice_agent_todo.graph as graph_module

    monkeypatch.setattr(graph_module, "create_agent", lambda model: FakeAgent())
    graph = build_todo_graph(FakeLlm(), checkpointer=InMemorySaver())

    state = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Today I need to develop the LangGraph voice assistant, "
                        "test realtime transcription, write the todo graph logic, "
                        "and update project notes."
                    )
                )
            ],
            "todo_list": [],
        },
        config={"configurable": {"thread_id": "today-plan"}},
    )

    assert state["action"] == "add_tasks_to_todo_list"


def test_update_status_handles_multiple_voice_updates_in_one_command(monkeypatch):
    import voice_agent_todo.graph as graph_module

    monkeypatch.setattr(graph_module, "create_agent", lambda model: FakeAgent())
    graph = build_todo_graph(FakeLlm(), checkpointer=InMemorySaver())

    initial_tasks = [
        {
            "task": "Write a blog about Retrieval Augmented Generation systems",
            "priority": 1,
            "completed": False,
            "active": True,
        },
        {
            "task": "dummy task",
            "priority": 2,
            "completed": False,
            "active": True,
        },
        {
            "task": "Find out the differences between Vercel AI SDK and LangChain LangGraph",
            "priority": 3,
            "completed": False,
            "active": True,
        },
    ]

    state = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Now, make the priority number one task completed. "
                        "That means make the status of completion to yes, "
                        "and then move on to the priority number two task, "
                        "which is the dummy task. Make the activation status to no."
                    )
                )
            ],
            "todo_list": initial_tasks,
        },
        config={"configurable": {"thread_id": "multi-status-voice"}} ,
    )

    assert state["action"] == "update_status"
    assert state["todo_list"][0]["completed"] is True
    assert state["todo_list"][0]["active"] is True
    assert state["todo_list"][1]["completed"] is False
    assert state["todo_list"][1]["active"] is False
    assert state["todo_list"][2]["completed"] is False
    assert state["todo_list"][2]["active"] is True


def test_update_status_handles_multiple_compact_status_changes(monkeypatch):
    import voice_agent_todo.graph as graph_module

    monkeypatch.setattr(graph_module, "create_agent", lambda model: FakeAgent())
    graph = build_todo_graph(FakeLlm(), checkpointer=InMemorySaver())

    initial_tasks = [
        {"task": "alpha task", "priority": 1, "completed": False, "active": True},
        {"task": "beta task", "priority": 2, "completed": False, "active": True},
    ]

    state = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Mark task 1 done, then set task 2 inactive.")
            ],
            "todo_list": initial_tasks,
        },
        config={"configurable": {"thread_id": "compact-multi-status"}},
    )

    assert state["todo_list"][0]["completed"] is True
    assert state["todo_list"][0]["active"] is True
    assert state["todo_list"][1]["completed"] is False
    assert state["todo_list"][1]["active"] is False
