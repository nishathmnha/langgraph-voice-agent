from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable

import sounddevice as sd
import websocket

from voice_agent_todo.config import DEFAULT_TRANSCRIPTION_PROMPT


class RealtimeTranscriber:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-transcribe",
        sample_rate: int = 24_000,
        silence_seconds: float = 5.0,
        max_record_seconds: int = 300,
        language: str | None = None,
        prompt: str | None = DEFAULT_TRANSCRIPTION_PROMPT,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.silence_seconds = silence_seconds
        self.max_record_seconds = max_record_seconds
        self.language = language
        self.prompt = prompt
        self.block_duration = 0.1
        self.block_size = int(self.sample_rate * self.block_duration)
        self.stop_event = threading.Event()

    def listen_once(self, on_final: Callable[[str], None] | None = None) -> str:
        final_transcript = ""
        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Safety-Identifier: local-voice-agent-user",
        ]

        def on_open(ws):
            transcription_config = {"model": self.model}
            if self.language:
                transcription_config["language"] = self.language
            if self.prompt:
                transcription_config["prompt"] = self.prompt

            ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": self.sample_rate},
                                    "transcription": transcription_config,
                                    "turn_detection": None,
                                }
                            },
                        },
                    }
                )
            )

            def stream_microphone():
                silent_chunks = 0
                total_chunks = 0
                speech_detected = False
                started_at = time.monotonic()

                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=self.block_size,
                ) as stream:
                    while not self.stop_event.is_set() and ws.sock and ws.sock.connected:
                        audio_chunk, _ = stream.read(self.block_size)
                        total_chunks += 1

                        rms = float((audio_chunk.astype("float32") ** 2).mean() ** 0.5)
                        if rms > 300:
                            speech_detected = True
                            silent_chunks = 0
                        elif speech_detected:
                            silent_chunks += 1

                        audio_base64 = base64.b64encode(audio_chunk.tobytes()).decode("utf-8")
                        ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_base64}))

                        silence_time = silent_chunks * self.block_duration
                        total_time = time.monotonic() - started_at

                        if speech_detected and silence_time >= self.silence_seconds:
                            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            return

                        if total_time >= self.max_record_seconds:
                            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            return

            threading.Thread(target=stream_microphone, daemon=True).start()

        def on_message(ws, message):
            nonlocal final_transcript
            event = json.loads(message)

            if event.get("type") == "conversation.item.input_audio_transcription.completed":
                final_transcript = event.get("transcript", "")
                if on_final:
                    on_final(final_transcript)
                self.stop_event.set()
                ws.close()

            if event.get("type") == "error":
                self.stop_event.set()
                ws.close()

        ws = websocket.WebSocketApp(url, header=headers, on_open=on_open, on_message=on_message)
        ws.run_forever(ping_interval=20, ping_timeout=10)
        return final_transcript
