# Voice Agent LangGraph Todo List

A small codebase extracted from `notebook/test_1.ipynb`.

## Ownership

Copyright (c) 2026 Ahamed Nishath. All rights reserved.

This repository is publicly viewable, but it is not open source. No permission
is granted to reuse, copy, modify, redistribute, deploy, or commercialize this
code without prior written permission from Ahamed Nishath.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the full ownership and usage terms.

It has:

- realtime voice transcription helper
- LangGraph todo workflow
- in-memory checkpointing per logged-in user
- simple text CLI for testing the graph

## Setup

```powershell
cd codebase\voice-agent-langgraph-todo-list
copy .env.example .env
```

Add your `OPENAI_API_KEY` to `.env`.

For the most accurate speech-to-text results, keep:

```text
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe
```

You can optionally set `OPENAI_TRANSCRIPTION_LANGUAGE` if you always speak one language, or leave it blank to auto-detect.

Install dependencies from the parent environment or create a new one:

```powershell
pip install -e .
```

## Run Text Mode

```powershell
voice-todo
```

## Run Simple Browser UI

From this folder:

Git Bash:

```bash
PYTHONPATH=src ../.venv/Scripts/python.exe -m voice_agent_todo.web_ui
```

PowerShell:

```powershell
..\.venv\Scripts\python.exe -m voice_agent_todo.web_ui
```

Then open:

```text
http://127.0.0.1:8765
```

The FastAPI UI lets you speak through the browser microphone, send transcript-style text, and inspect the todo list table.

The browser UI streams live partial transcript updates while you speak, then runs a final higher-accuracy transcription pass before the text is shown as complete.

Type commands like:

```text
Today I need to develop the LangGraph voice assistant, test realtime transcription, and update project notes.
Add these tasks: create a README, prepare the demo script, and clean the notebook.
Change the project notes task to write detailed architecture notes.
Remove the project notes task.
Mark the README task done.
```

## Use From Python

```python
from langchain_core.messages import HumanMessage
from voice_agent_todo.app import TodoAssistant

assistant = TodoAssistant()

state = assistant.invoke([
    HumanMessage(content="Add tasks: create a README and clean the notebook.")
])

print(state["todo_list"])
```

## Memory

The app uses `InMemorySaver` and a thread id based on the logged-in OS user:

```python
todo-assistant-<username>
```

This memory lasts only while the Python process is alive.
