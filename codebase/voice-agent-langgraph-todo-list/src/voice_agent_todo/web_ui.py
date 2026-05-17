from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import threading
import traceback
import wave
import webbrowser
from typing import Any

import httpx
from openai import OpenAI
import uvicorn
import websocket as realtime_websocket
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from voice_agent_todo.app import TodoAssistant
from voice_agent_todo.config import get_settings
import voice_agent_todo.graph as graph_module


PROXY_ENV_VARS = (
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
)


def clear_proxy_env_for_openai() -> dict[str, str]:
    previous_values = {}
    for name in PROXY_ENV_VARS:
        if name in os.environ:
            previous_values[name] = os.environ.pop(name)
    return previous_values


def restore_proxy_env(previous_values: dict[str, str]) -> None:
    for name, value in previous_values.items():
        os.environ[name] = value


def build_transcription_request(settings) -> dict[str, Any]:
    request: dict[str, Any] = {"model": settings.transcription_model}
    if settings.transcription_language:
        request["language"] = settings.transcription_language
    if settings.transcription_prompt:
        request["prompt"] = settings.transcription_prompt
    return request


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24_000) -> bytes:
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return wav_buffer.getvalue()


def transcribe_audio_bytes(audio_file: Any, settings) -> str:
    previous_proxy_values = clear_proxy_env_for_openai()
    try:
        client = OpenAI(
            api_key=settings.openai_api_key,
            http_client=httpx.Client(trust_env=False, timeout=60),
        )
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            **build_transcription_request(settings),
        )
    finally:
        restore_proxy_env(previous_proxy_values)

    return (getattr(transcript, "text", "") or "").strip()


def transcribe_pcm_audio(pcm_bytes: bytes, settings, sample_rate: int = 24_000) -> str:
    if not pcm_bytes:
        return ""

    wav_bytes = pcm16_to_wav_bytes(pcm_bytes, sample_rate=sample_rate)
    audio_file = ("voice-input.wav", wav_bytes, "audio/wav")
    return transcribe_audio_bytes(audio_file, settings)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="author" content="Ahamed Nishath" />
  <meta name="copyright" content="Ahamed Nishath" />
  <title>Voice Todo Assistant</title>
  <style>
    :root {
      --bg: #0f1115;
      --bg-soft: #151922;
      --panel: #171b24;
      --panel-strong: #1d2230;
      --text: #edf1f7;
      --muted: #98a2b3;
      --line: rgba(255, 255, 255, 0.08);
      --accent: #8fb8ff;
      --accent-strong: #6ca3ff;
      --accent-soft: rgba(143, 184, 255, 0.12);
      --danger: #ff7b72;
      --danger-soft: rgba(255, 123, 114, 0.12);
      --success: #7fd1a5;
      --success-soft: rgba(127, 209, 165, 0.12);
      --inactive: #202633;
      --shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
      --radius-xl: 18px;
      --radius-lg: 16px;
      --radius-md: 12px;
      --radius-sm: 10px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(180deg, #0d1016 0%, #11151d 100%);
    }

    body::before {
      display: none;
    }

    main {
      position: relative;
      z-index: 1;
      max-width: 1220px;
      margin: 0 auto;
      padding: 36px 20px 40px;
    }

    .hero {
      display: block;
      margin-bottom: 18px;
    }

    .hero-card,
    .composer,
    .task-shell,
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }

    .hero-card {
      border-radius: var(--radius-xl);
      padding: 24px 26px;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--accent);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    h1 {
      margin: 16px 0 10px;
      font-size: clamp(30px, 4vw, 40px);
      line-height: 1.08;
      letter-spacing: -0.03em;
      font-weight: 700;
    }

    .sub {
      max-width: 56ch;
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
    }

    .hero-list {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 20px;
    }

    .hero-chip {
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }

    .stack-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }

    .stack-card {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      background: #11151d;
      border: 1px solid var(--line);
    }

    .stack-icon {
      width: 30px;
      height: 30px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 9px;
      background: rgba(143, 184, 255, 0.12);
      color: var(--accent);
      flex: 0 0 auto;
    }

    .stack-icon svg {
      width: 16px;
      height: 16px;
      display: block;
    }

    .stack-copy {
      display: grid;
      gap: 2px;
      min-width: 0;
    }

    .stack-name {
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
      letter-spacing: 0.01em;
    }

    .stack-role {
      font-size: 11px;
      color: var(--muted);
      line-height: 1.35;
    }

    .hero-credit {
      margin-top: 16px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    .hero-credit strong {
      color: var(--text);
      font-weight: 600;
    }

    .workspace {
      display: block;
      align-items: start;
    }

    .composer {
      border-radius: var(--radius-xl);
      padding: 20px;
      display: grid;
      gap: 14px;
    }

    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .section-title h2,
    .control-title,
    .task-shell h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .section-hint,
    .control-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    textarea,
    input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #11151d;
      color: var(--text);
      font: inherit;
      outline: none;
      transition: border-color 120ms ease, box-shadow 120ms ease;
    }

    textarea {
      min-height: 148px;
      resize: vertical;
      padding: 16px 18px;
      line-height: 1.55;
    }

    input[type="text"] {
      min-height: 48px;
      padding: 0 14px;
    }

    textarea:focus,
    input[type="text"]:focus {
      border-color: rgba(143, 184, 255, 0.45);
      box-shadow: 0 0 0 3px rgba(143, 184, 255, 0.08);
    }

    .actions {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }

    .button-row,
    .mini-button-row,
    .filter-row,
    .bulk-actions,
    .manual-row-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    button {
      border: 0;
      border-radius: 10px;
      padding: 11px 16px;
      background: var(--accent);
      color: #0f1115;
      font: inherit;
      font-weight: 700;
      letter-spacing: -0.01em;
      cursor: pointer;
      transition: background 120ms ease, opacity 120ms ease, border-color 120ms ease;
      box-shadow: none;
    }

    button:hover:not(:disabled) {
      background: var(--accent-strong);
    }

    button.secondary {
      background: #202633;
      color: var(--text);
    }

    button.ghost {
      background: transparent;
      color: var(--accent);
      border: 1px solid var(--line);
    }

    button.voice {
      background: #edf1f7;
      color: #0f1115;
    }

    button.voice.listening {
      background: var(--danger);
      color: #0f1115;
    }

    button.warn {
      background: #313949;
      color: var(--text);
    }

    button.success {
      background: #7fd1a5;
      color: #0f1115;
    }

    button.small {
      padding: 9px 12px;
      font-size: 12px;
      border-radius: 9px;
    }

    button:disabled {
      opacity: 0.55;
      cursor: wait;
      transform: none;
      box-shadow: none;
    }

    .status {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }

    .status strong {
      color: var(--text);
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 6px;
    }

    .metric {
      border-radius: var(--radius-lg);
      padding: 16px 18px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
      font-weight: 700;
    }

    .metric strong {
      font-size: 28px;
      line-height: 1;
      letter-spacing: -0.03em;
      display: block;
      min-width: 0;
    }

    .metric strong.metric-action {
      font-size: clamp(18px, 2vw, 28px);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .filter-row button.active {
      background: var(--accent);
      color: #0f1115;
    }

    .task-shell {
      border-radius: var(--radius-xl);
      padding: 18px 18px 14px;
      margin-top: 18px;
      overflow: hidden;
    }

    .task-header {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }

    .table-wrap {
      overflow-x: auto;
      border-radius: var(--radius-lg);
      border: 1px solid var(--line);
      background: #131823;
    }

    table {
      width: 100%;
      border-collapse: collapse;
    }

    th, td {
      padding: 15px 16px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }

    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: #171d29;
    }

    tr:last-child td { border-bottom: 0; }
    tr.inactive { background: #151922; color: var(--muted); }
    tr.done { background: rgba(127, 209, 165, 0.06); }

    .priority-pill,
    .state-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 32px;
      min-width: 32px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }

    .priority-pill {
      background: rgba(143, 184, 255, 0.12);
      color: var(--accent);
    }

    .state-pill {
      background: #202633;
      color: var(--text);
      min-width: auto;
      font-weight: 700;
    }

    .state-pill.done {
      background: rgba(127, 209, 165, 0.12);
      color: var(--success);
    }

    .state-pill.inactive {
      background: rgba(152, 162, 179, 0.12);
      color: var(--muted);
    }

    .task-text {
      display: grid;
      gap: 8px;
    }

    .task-name {
      font-size: 16px;
      line-height: 1.45;
      font-weight: 600;
      color: var(--text);
    }

    .manual-row-actions {
      margin-top: 2px;
    }

    .empty {
      background: #131823;
      border: 1px dashed var(--line);
      border-radius: var(--radius-lg);
      padding: 42px 24px;
      color: var(--muted);
      text-align: center;
      font-size: 15px;
      line-height: 1.6;
    }

    .debug {
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
      word-break: break-word;
    }

    .site-footer {
      margin-top: 16px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .site-footer strong {
      color: var(--text);
      font-weight: 600;
    }

    @media (max-width: 760px) {
      main {
        padding: 20px 14px 28px;
      }

      .hero-card,
      .composer,
      .task-shell {
        padding: 18px;
      }

      .meta {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      th:nth-child(3),
      td:nth-child(3),
      th:nth-child(4),
      td:nth-child(4) {
        display: none;
      }
    }

    @media (max-width: 520px) {
      .meta {
        grid-template-columns: 1fr;
      }

      .button-row,
      .mini-button-row,
      .filter-row,
      .bulk-actions,
      .manual-row-actions {
        gap: 8px;
      }

      button {
        width: 100%;
        justify-content: center;
      }

      .site-footer {
        flex-direction: column;
        align-items: flex-start;
      }

      .task-header,
      .actions,
      .section-title {
        align-items: flex-start;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">AI Task Orchestration</div>
        <h1>Voice-Controlled Todo Assistant</h1>
        <p class="sub">
          Manage tasks with natural voice commands, fast status updates, and a clear real-time view of your working list.
        </p>
        <div class="hero-list">
          <div class="hero-chip">Live transcription</div>
          <div class="hero-chip">Multi-step task updates</div>
          <div class="hero-chip">Manual task controls</div>
        </div>
        <div class="stack-strip" aria-label="Technology stack">
          <div class="stack-card">
            <div class="stack-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M5 8l7-4 7 4v8l-7 4-7-4V8z"></path>
                <path d="M12 4v16"></path>
                <path d="M5 8l7 4 7-4"></path>
              </svg>
            </div>
            <div class="stack-copy">
              <div class="stack-name">LangGraph</div>
              <div class="stack-role">Workflow orchestration</div>
            </div>
          </div>
          <div class="stack-card">
            <div class="stack-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 3l7 4v10l-7 4-7-4V7l7-4z"></path>
                <path d="M9 9h6v6H9z"></path>
              </svg>
            </div>
            <div class="stack-copy">
              <div class="stack-name">Python</div>
              <div class="stack-role">Application runtime</div>
            </div>
          </div>
          <div class="stack-card">
            <div class="stack-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 7h16"></path>
                <path d="M4 12h10"></path>
                <path d="M4 17h7"></path>
                <path d="M17 10l3 2-3 2"></path>
              </svg>
            </div>
            <div class="stack-copy">
              <div class="stack-name">FastAPI</div>
              <div class="stack-role">API and UI delivery</div>
            </div>
          </div>
          <div class="stack-card">
            <div class="stack-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 10a8 8 0 0 1 16 0"></path>
                <path d="M6 14a6 6 0 0 1 12 0"></path>
                <path d="M8 18a4 4 0 0 1 8 0"></path>
                <circle cx="12" cy="18" r="1"></circle>
              </svg>
            </div>
            <div class="stack-copy">
              <div class="stack-name">OpenAI Realtime</div>
              <div class="stack-role">Live voice transcription</div>
            </div>
          </div>
          <div class="stack-card">
            <div class="stack-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 7h8a3 3 0 0 1 3 3v7"></path>
                <path d="M20 17h-8a3 3 0 0 1-3-3V7"></path>
                <path d="M15 5l4 2-4 2"></path>
                <path d="M9 19l-4-2 4-2"></path>
              </svg>
            </div>
            <div class="stack-copy">
              <div class="stack-name">WebSockets</div>
              <div class="stack-role">Streaming audio events</div>
            </div>
          </div>
        </div>
        <div class="hero-credit">Created and presented by <strong>Ahamed Nishath</strong></div>
      </div>

    </section>

    <section class="workspace">
      <div>
        <section class="composer">
          <div class="section-title">
            <div>
              <h2>Command Composer</h2>
              <div class="section-hint">Voice and text both route through the same assistant.</div>
            </div>
          </div>

          <textarea id="message">Today I need to develop the LangGraph voice assistant, test realtime transcription, write the todo graph logic, and update project notes.</textarea>

          <div class="actions">
            <div class="button-row">
              <button id="voice" class="voice" type="button">Start Voice Capture</button>
              <button id="send" type="button">Send Command</button>
              <button id="clearComposer" class="secondary" type="button">Clear Draft</button>
              <button id="clear" class="ghost" type="button">Clear Thread</button>
            </div>
            <span id="status" class="status"><strong>Ready</strong> for voice or typed input.</span>
          </div>

          <section class="meta">
            <div class="metric"><span>Last Action</span><strong id="action" class="metric-action">-</strong></div>
            <div class="metric"><span>Active Tasks</span><strong id="activeCount">0</strong></div>
            <div class="metric"><span>Completed</span><strong id="completedCount">0</strong></div>
            <div class="metric"><span>Total Tasks</span><strong id="totalCount">0</strong></div>
          </section>
        </section>

        <section class="task-shell">
          <div class="task-header">
            <div>
              <h2>Todo Board</h2>
              <div class="section-hint">Use the filters and per-row buttons for direct manual control.</div>
            </div>
            <div class="filter-row" id="filterRow">
              <button type="button" class="secondary active" data-filter="all">All</button>
              <button type="button" class="secondary" data-filter="active">Active</button>
              <button type="button" class="secondary" data-filter="done">Completed</button>
              <button type="button" class="secondary" data-filter="inactive">Inactive</button>
            </div>
          </div>

          <section id="tableWrap" class="empty">No tasks yet. Add a task, speak a command, or use the quick controls to get started.</section>
          <div id="debug" class="debug">Debug: waiting for request.</div>
        </section>
        <footer class="site-footer">
          <div>© <span id="copyrightYear"></span> <strong>Ahamed Nishath</strong>. All rights reserved.</div>
          <div>Voice-Controlled Todo Assistant demo interface.</div>
        </footer>
      </div>
    </section>
  </main>

  <script>
    const message = document.querySelector("#message");
    const voice = document.querySelector("#voice");
    const send = document.querySelector("#send");
    const clear = document.querySelector("#clear");
    const clearComposer = document.querySelector("#clearComposer");
    const statusEl = document.querySelector("#status");
    const actionEl = document.querySelector("#action");
    const activeCount = document.querySelector("#activeCount");
    const completedCount = document.querySelector("#completedCount");
    const totalCount = document.querySelector("#totalCount");
    const tableWrap = document.querySelector("#tableWrap");
    const debug = document.querySelector("#debug");
    const copyrightYear = document.querySelector("#copyrightYear");
    const filterRow = document.querySelector("#filterRow");

    let streamSocket = null;
    let micStream = null;
    let audioContext = null;
    let sourceNode = null;
    let processorNode = null;
    let listening = false;
    let committed = false;
    let finalTranscript = "";
    let liveTranscript = "";
    let currentState = { todo_list: [], debug: {} };
    let activeFilter = "all";
    const TARGET_SAMPLE_RATE = 24000;

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function formatActionLabel(action) {
      const labels = {
        add_tasks_to_todo_list: "Add Tasks",
        update_task: "Update Task",
        update_status: "Update Status",
      };
      return labels[action] || action || "-";
    }

    function statusText(text, strongText) {
      statusEl.innerHTML = strongText ? `<strong>${escapeHtml(strongText)}</strong> ${escapeHtml(text)}` : escapeHtml(text);
    }

    function getFilteredTasks(tasks) {
      if (activeFilter === "active") return tasks.filter(task => task.active !== false);
      if (activeFilter === "done") return tasks.filter(task => task.completed === true);
      if (activeFilter === "inactive") return tasks.filter(task => task.active === false);
      return tasks;
    }

    function setFilter(filter) {
      activeFilter = filter;
      [...filterRow.querySelectorAll("[data-filter]")].forEach(button => {
        button.classList.toggle("active", button.dataset.filter === filter);
      });
      render(currentState);
    }

    function buildTaskCommand(task, action) {
      const priority = task.priority;
      if (action === "done") return `Mark priority ${priority} task done.`;
      if (action === "reopen") return `Reopen priority ${priority} task.`;
      if (action === "deactivate") return `Set priority ${priority} task inactive.`;
      if (action === "activate") return `Restore priority ${priority} task.`;
      return "";
    }

    function render(state) {
      currentState = state || { todo_list: [], debug: {} };
      const tasks = currentState.todo_list || [];
      const filteredTasks = getFilteredTasks(tasks);
      const completedTasks = tasks.filter(task => task.completed === true).length;
      const activeTasks = tasks.filter(task => task.active !== false).length;

      actionEl.textContent = formatActionLabel(currentState.action);
      actionEl.title = currentState.action || "-";
      activeCount.textContent = activeTasks;
      completedCount.textContent = completedTasks;
      totalCount.textContent = tasks.length;

      const graphFile = currentState.debug?.graph_file || "-";
      debug.textContent = `Debug: action=${currentState.action || "-"} response=${currentState.response || "-"} tasks=${tasks.length} graph=${graphFile}`;

      if (!tasks.length) {
        tableWrap.className = "empty";
        tableWrap.innerHTML = "No tasks yet. Add a task, speak a command, or use the quick controls to get started.";
        return;
      }

      if (!filteredTasks.length) {
        tableWrap.className = "empty";
        tableWrap.innerHTML = `No tasks match the <strong>${escapeHtml(activeFilter)}</strong> filter right now.`;
        return;
      }

      tableWrap.className = "table-wrap";
      tableWrap.innerHTML = `
        <table>
          <thead>
            <tr>
              <th style="width: 104px;">Priority</th>
              <th>Task</th>
              <th style="width: 140px;">Completed</th>
              <th style="width: 120px;">Active</th>
            </tr>
          </thead>
          <tbody>
            ${filteredTasks.map(task => {
              const inactive = task.active === false;
              const done = task.completed === true;
              const rowClass = inactive ? "inactive" : done ? "done" : "";
              return `
                <tr class="${rowClass}">
                  <td><span class="priority-pill">${escapeHtml(task.priority)}</span></td>
                  <td>
                    <div class="task-text">
                      <div class="task-name">${escapeHtml(task.task)}</div>
                      <div class="manual-row-actions">
                        ${done
                          ? `<button type="button" class="small secondary row-action" data-priority="${task.priority}" data-action="reopen">Reopen</button>`
                          : `<button type="button" class="small success row-action" data-priority="${task.priority}" data-action="done">Mark Done</button>`
                        }
                        ${inactive
                          ? `<button type="button" class="small secondary row-action" data-priority="${task.priority}" data-action="activate">Restore</button>`
                          : `<button type="button" class="small ghost row-action" data-priority="${task.priority}" data-action="deactivate">Deactivate</button>`
                        }
                        <button type="button" class="small secondary row-insert" data-command="${escapeHtml(buildTaskCommand(task, done ? "reopen" : "done"))}">Stage Command</button>
                      </div>
                    </div>
                  </td>
                  <td><span class="state-pill ${done ? "done" : ""}">${done ? "Yes" : "No"}</span></td>
                  <td><span class="state-pill ${inactive ? "inactive" : ""}">${inactive ? "No" : "Yes"}</span></td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      `;
    }

    async function postJson(url, body = {}) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Request failed");
      return data;
    }

    async function sendCommandText(text, options = {}) {
      const { keepDraft = false, pendingLabel = "Working..." } = options;
      const trimmed = String(text || "").trim();
      if (!trimmed) return;

      send.disabled = true;
      statusText(pendingLabel, "Working.");
      try {
        const state = await postJson("/api/message", { message: trimmed });
        render(state);
        statusText(state.response || "Done.", "Updated.");
        if (!keepDraft) {
          message.value = "";
        }
      } catch (error) {
        statusText(error.message, "Error.");
      } finally {
        send.disabled = false;
      }
    }

    async function setupVoice() {
      if (!navigator.mediaDevices || !window.AudioContext) {
        voice.disabled = true;
        voice.textContent = "Mic Unsupported";
        statusText("Voice input needs microphone streaming support.", "Notice.");
        return;
      }
    }

    function arrayBufferToBase64(buffer) {
      let binary = "";
      const bytes = new Uint8Array(buffer);
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      return btoa(binary);
    }

    function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
      if (outputSampleRate === inputSampleRate) return buffer;
      const sampleRateRatio = inputSampleRate / outputSampleRate;
      const newLength = Math.round(buffer.length / sampleRateRatio);
      const result = new Float32Array(newLength);
      let offsetResult = 0;
      let offsetBuffer = 0;

      while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accumulator = 0;
        let count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
          accumulator += buffer[i];
          count += 1;
        }
        result[offsetResult] = accumulator / Math.max(count, 1);
        offsetResult += 1;
        offsetBuffer = nextOffsetBuffer;
      }
      return result;
    }

    function floatTo16BitPcm(float32Array) {
      const buffer = new ArrayBuffer(float32Array.length * 2);
      const view = new DataView(buffer);
      for (let i = 0; i < float32Array.length; i++) {
        const value = Math.max(-1, Math.min(1, float32Array[i]));
        view.setInt16(i * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
      }
      return buffer;
    }

    function updateLiveTranscript() {
      message.value = `${finalTranscript} ${liveTranscript}`.trim();
    }

    function cleanupAudio() {
      if (processorNode) processorNode.disconnect();
      if (sourceNode) sourceNode.disconnect();
      if (audioContext) audioContext.close().catch(() => {});
      if (micStream) micStream.getTracks().forEach(track => track.stop());
      processorNode = null;
      sourceNode = null;
      audioContext = null;
      micStream = null;
    }

    function finishStreaming(statusMessage = "Finalizing live transcript...") {
      if (!listening || committed) return;
      committed = true;
      listening = false;
      voice.disabled = true;
      voice.classList.remove("listening");
      voice.textContent = "Finalizing...";
      statusText(statusMessage, "Voice.");
      cleanupAudio();

      if (streamSocket && streamSocket.readyState === WebSocket.OPEN) {
        streamSocket.send(JSON.stringify({ type: "commit" }));
      }
    }

    async function startStreaming() {
      committed = false;
      finalTranscript = "";
      liveTranscript = "";
      message.value = "";

      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      streamSocket = new WebSocket(`${scheme}://${window.location.host}/ws/transcribe`);

      streamSocket.onmessage = event => {
        const payload = JSON.parse(event.data);
        if (payload.type === "ready") {
          statusText("Streaming now. Speak naturally.", "Voice.");
        } else if (payload.type === "delta") {
          liveTranscript += payload.text || "";
          updateLiveTranscript();
        } else if (payload.type === "completed") {
          finalTranscript = payload.text || message.value || "";
          liveTranscript = "";
          updateLiveTranscript();
          statusText(finalTranscript ? "Transcript captured. Review or send it." : "No speech detected.", "Voice.");
          voice.disabled = false;
          voice.textContent = "Start Voice Capture";
          streamSocket.close();
        } else if (payload.type === "error") {
          statusText(payload.message || "Transcription error", "Voice error.");
          voice.disabled = false;
          voice.textContent = "Start Voice Capture";
          cleanupAudio();
        }
      };

      streamSocket.onclose = () => {
        cleanupAudio();
        listening = false;
        voice.classList.remove("listening");
        voice.disabled = false;
        voice.textContent = "Start Voice Capture";
      };

      await new Promise((resolve, reject) => {
        streamSocket.onopen = resolve;
        streamSocket.onerror = () => reject(new Error("Could not connect to live transcription."));
      });

      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: TARGET_SAMPLE_RATE,
          noiseSuppression: true,
          echoCancellation: true,
          autoGainControl: true,
        }
      });
      audioContext = new AudioContext();
      sourceNode = audioContext.createMediaStreamSource(micStream);
      processorNode = audioContext.createScriptProcessor(4096, 1, 1);

      processorNode.onaudioprocess = event => {
        if (!listening || !streamSocket || streamSocket.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const downsampled = downsampleBuffer(input, audioContext.sampleRate, TARGET_SAMPLE_RATE);
        const pcmBuffer = floatTo16BitPcm(downsampled);
        streamSocket.send(JSON.stringify({
          type: "audio",
          audio: arrayBufferToBase64(pcmBuffer),
        }));
      };

      sourceNode.connect(processorNode);
      processorNode.connect(audioContext.destination);
      listening = true;
      voice.classList.add("listening");
      voice.textContent = "Stop Voice Capture";
      statusText("Streaming now. Speak naturally.", "Voice.");
    }

    voice.addEventListener("click", async () => {
      if (listening) {
        finishStreaming();
      } else {
        try {
          await startStreaming();
        } catch (error) {
          statusText(error.message, "Voice error.");
          cleanupAudio();
          voice.disabled = false;
          voice.textContent = "Start Voice Capture";
        }
      }
    });

    send.addEventListener("click", async () => {
      await sendCommandText(message.value, { pendingLabel: "Sending command to the assistant..." });
    });

    clearComposer.addEventListener("click", () => {
      message.value = "";
      statusText("Draft cleared. Your current thread is unchanged.", "Ready.");
    });

    clear.addEventListener("click", async () => {
      clear.disabled = true;
      statusText("Clearing the current thread...", "Working.");
      try {
        const state = await postJson("/api/clear");
        render(state);
        statusText("Thread cleared.", "Updated.");
      } catch (error) {
        statusText(error.message, "Error.");
      } finally {
        clear.disabled = false;
      }
    });

    filterRow.addEventListener("click", event => {
      const button = event.target.closest("[data-filter]");
      if (!button) return;
      setFilter(button.dataset.filter);
    });

    document.addEventListener("click", async event => {
      const rowAction = event.target.closest(".row-action");
      if (rowAction) {
        const priority = rowAction.dataset.priority;
        const action = rowAction.dataset.action;
        const task = (currentState.todo_list || []).find(item => String(item.priority) === String(priority));
        if (!task) return;
        await sendCommandText(buildTaskCommand(task, action), { pendingLabel: `Updating task ${priority}...` });
        return;
      }

      const rowInsert = event.target.closest(".row-insert");
      if (rowInsert) {
        message.value = rowInsert.dataset.command || "";
        message.focus();
        statusText("Command staged in the composer. Review it or send it now.", "Ready.");
        return;
      }

    });

    setFilter("all");
    if (copyrightYear) {
      copyrightYear.textContent = new Date().getFullYear();
    }
    setupVoice();
    render(currentState);
  </script>
</body>
</html>
"""


class MessageRequest(BaseModel):
    message: str


class TranscribeRequest(BaseModel):
    audio_base64: str
    mime_type: str = "audio/webm"


class UiState:
    def __init__(self) -> None:
        self.assistant = TodoAssistant()

    def reset(self) -> None:
        self.assistant = TodoAssistant(thread_id=self.assistant.thread_id)


ui_state = UiState()
app = FastAPI(title="Voice Todo Assistant")


@app.websocket("/ws/transcribe")
async def stream_transcription(client_ws: WebSocket) -> None:
    await client_ws.accept()

    settings = get_settings()
    loop = asyncio.get_running_loop()
    done = asyncio.Event()
    recorded_pcm = bytearray()
    realtime_url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = [
        f"Authorization: Bearer {settings.openai_api_key}",
        "OpenAI-Safety-Identifier: local-web-ui-user",
    ]

    def open_realtime_socket():
        previous_proxy_values = clear_proxy_env_for_openai()
        try:
            ws = realtime_websocket.create_connection(
                realtime_url,
                header=headers,
                timeout=20,
                http_proxy_host=None,
                http_proxy_port=None,
            )
            ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24000},
                                    "transcription": build_transcription_request(settings),
                                    "turn_detection": None,
                                }
                            },
                        },
                    }
                )
            )
            return ws
        finally:
            restore_proxy_env(previous_proxy_values)

    try:
        openai_ws = await asyncio.to_thread(open_realtime_socket)
        print("[web-ui] realtime transcription connected", flush=True)
        await client_ws.send_json({"type": "ready"})
    except Exception as exc:
        print("[web-ui] realtime transcription connection failed", flush=True)
        traceback.print_exc()
        await client_ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        await client_ws.close()
        return

    def send_to_client(payload: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(client_ws.send_json(payload), loop)

    def read_openai_events() -> None:
        try:
            while not done.is_set():
                raw_message = openai_ws.recv()
                if not raw_message:
                    continue
                event = json.loads(raw_message)
                event_type = event.get("type")

                if event_type == "conversation.item.input_audio_transcription.delta":
                    send_to_client({"type": "delta", "text": event.get("delta", "")})
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    realtime_text = (event.get("transcript", "") or "").strip()
                    final_text = realtime_text

                    try:
                        refined_text = transcribe_pcm_audio(bytes(recorded_pcm), settings)
                        if refined_text:
                            final_text = refined_text
                    except Exception:
                        print("[web-ui] high-accuracy final transcription failed", flush=True)
                        traceback.print_exc()

                    send_to_client(
                        {
                            "type": "completed",
                            "text": final_text,
                            "realtime_text": realtime_text,
                        }
                    )
                    loop.call_soon_threadsafe(done.set)
                    return
                elif event_type == "error":
                    error = event.get("error", {})
                    send_to_client({"type": "error", "message": error.get("message", str(event))})
                    loop.call_soon_threadsafe(done.set)
                    return
        except Exception as exc:
            if not done.is_set():
                print("[web-ui] realtime reader failed", flush=True)
                traceback.print_exc()
                send_to_client({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                loop.call_soon_threadsafe(done.set)

    reader = threading.Thread(target=read_openai_events, daemon=True)
    reader.start()

    try:
        while not done.is_set():
            try:
                message_data = await asyncio.wait_for(client_ws.receive_json(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            message_type = message_data.get("type")
            if message_type == "audio":
                audio_base64 = message_data.get("audio", "")
                try:
                    recorded_pcm.extend(base64.b64decode(audio_base64, validate=True))
                except Exception:
                    pass
                openai_ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": audio_base64,
                        }
                    )
                )
            elif message_type == "commit":
                openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            elif message_type == "close":
                loop.call_soon_threadsafe(done.set)
                break
    except WebSocketDisconnect:
        loop.call_soon_threadsafe(done.set)
    except Exception:
        print("[web-ui] browser websocket failed", flush=True)
        traceback.print_exc()
        loop.call_soon_threadsafe(done.set)
    finally:
        done.set()
        try:
            openai_ws.close()
        except Exception:
            pass


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in state.get("messages", []):
        messages.append(getattr(message, "content", str(message)))

    return {
        "action": state.get("action"),
        "response": state.get("response"),
        "todo_list": state.get("todo_list", []),
        "messages": messages,
        "debug": {
            "graph_file": graph_module.__file__,
            "task_count": len(state.get("todo_list", [])),
        },
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/api/message")
def send_message(request: MessageRequest) -> dict[str, Any]:
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message is empty.")

    try:
        state = ui_state.assistant.invoke(text)
        print(
            "[web-ui] action={action} total_tasks={count} graph_file={graph_file}".format(
                action=state.get("action"),
                count=len(state.get("todo_list", [])),
                graph_file=graph_module.__file__,
            ),
            flush=True,
        )
        return serialize_state(dict(state))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/transcribe")
def transcribe_audio(request: TranscribeRequest) -> dict[str, str]:
    try:
        settings = get_settings()
        audio_bytes = base64.b64decode(request.audio_base64, validate=True)
        if len(audio_bytes) < 1000:
            raise ValueError("Recorded audio is too short. Please speak for a moment and try again.")

        mime_type = (request.mime_type or "audio/webm").split(";")[0]
        extension = "webm"
        if "mp4" in mime_type:
            extension = "mp4"
        elif "wav" in mime_type:
            extension = "wav"
        elif "ogg" in mime_type:
            extension = "ogg"
        elif "mpeg" in mime_type or "mp3" in mime_type:
            extension = "mp3"

        audio_file = (f"voice-input.{extension}", audio_bytes, mime_type)
        print(
            f"[web-ui] transcribe request bytes={len(audio_bytes)} mime={request.mime_type} model={settings.transcription_model}",
            flush=True,
        )

        text = transcribe_audio_bytes(audio_file, settings)
        print(f"[web-ui] transcription={text}", flush=True)
        return {"text": text}
    except Exception as exc:
        print("[web-ui] transcription failed", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/clear")
def clear_thread() -> dict[str, Any]:
    ui_state.reset()
    return {
        "todo_list": [],
        "action": None,
        "response": "Thread cleared.",
        "messages": [],
        "debug": {"graph_file": graph_module.__file__, "task_count": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"FastAPI UI running at {url}")
    print(f"Thread: {ui_state.assistant.thread_id}")

    if not args.no_open:
        webbrowser.open(url)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
