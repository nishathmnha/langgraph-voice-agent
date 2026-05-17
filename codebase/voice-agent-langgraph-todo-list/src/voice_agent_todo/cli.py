from __future__ import annotations

from pprint import pprint

from voice_agent_todo.app import TodoAssistant


def main() -> None:
    assistant = TodoAssistant()
    print("Voice todo assistant text mode. Type 'exit' to quit.")
    print(f"Thread: {assistant.thread_id}")

    while True:
        user_text = input("\nTask> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        state = assistant.invoke(user_text)
        pprint(
            {
                "action": state.get("action"),
                "response": state.get("response"),
                "todo_list": state.get("todo_list", []),
            }
        )


if __name__ == "__main__":
    main()
