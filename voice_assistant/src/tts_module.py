"""Text-to-speech module that integrates HiggsAudio."""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError as exc:  # pragma: no cover - requests should be installed with requirements
    raise RuntimeError("`requests` is required for the TTS client. Install with `pip install requests`.") from exc

logger = logging.getLogger(__name__)


@dataclass
class TTSResponse:
    audio_path: Path
    format: str
    sample_rate: int


class HiggsAudioTTS:
    """Minimal client for a HiggsAudio text-to-speech server."""

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        default_voice: str = "default",
        sample_rate: int = 22050,
        audio_format: str = "wav",
        timeout: int = 60,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.timeout = timeout
        logger.debug("Initialized HiggsAudioTTS with base=%s", self.api_base)

    def synthesize(
        self,
        text: str,
        output_dir: str | Path,
        *,
        voice: Optional[str] = None,
        stream: bool = False,
    ) -> TTSResponse:
        if not text.strip():
            raise ValueError("Text-to-speech input must not be empty")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"tts_{abs(hash(text))}.{self.audio_format}"

        payload = {
            "text": text,
            "voice": voice or self.default_voice,
            "sample_rate": self.sample_rate,
            "format": self.audio_format,
            "stream": stream,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_base}/tts"
        logger.debug("Requesting TTS voice=%s stream=%s", payload["voice"], stream)
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"HiggsAudio TTS request failed with {response.status_code}: {response.text}")

        data = response.json()
        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            raise RuntimeError("TTS response missing `audio_base64` field")

        audio_bytes = base64.b64decode(audio_b64)
        output_path.write_bytes(audio_bytes)
        logger.info("Generated speech audio at %s", output_path)
        return TTSResponse(audio_path=output_path, format=self.audio_format, sample_rate=self.sample_rate)

