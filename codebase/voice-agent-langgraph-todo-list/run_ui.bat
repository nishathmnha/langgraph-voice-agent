@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=src
"..\.venv\Scripts\python.exe" -m uvicorn voice_agent_todo.web_ui:app --host 127.0.0.1 --port 8765 --app-dir src --log-level debug
