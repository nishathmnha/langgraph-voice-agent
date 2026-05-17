from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe the speaker exactly. Preserve technical terms, product names, file names, "
    "numbers, and short task phrases when clear. Do not invent words that were not spoken."
)


def load_project_env() -> None:
    package_file = Path(__file__).resolve()
    project_root = package_file.parents[2]
    codebase_root = package_file.parents[3]

    for env_path in (
        project_root / ".env",
        codebase_root / ".env",
        codebase_root / "notebook" / ".env",
        Path(".env"),
        Path("../.env"),
        Path("../../.env"),
        Path("codebase/.env"),
    ):
        if env_path.exists():
            load_dotenv(env_path)
            return

    load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str = "gpt-5.4-mini"
    transcription_model: str = "gpt-4o-transcribe"
    transcription_language: str | None = None
    transcription_prompt: str = DEFAULT_TRANSCRIPTION_PROMPT


def get_settings() -> Settings:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to .env.")

    return Settings(
        openai_api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        transcription_model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe"),
        transcription_language=os.getenv("OPENAI_TRANSCRIPTION_LANGUAGE") or None,
        transcription_prompt=os.getenv("OPENAI_TRANSCRIPTION_PROMPT", DEFAULT_TRANSCRIPTION_PROMPT),
    )
