"""Speech recognition module powered by OpenAI Whisper."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    """Transcribed segment with time alignment."""

    text: str
    start: float
    end: float


@dataclass
class ASRResult:
    """Container for a transcription result."""

    text: str
    segments: List[ASRSegment]
    language: Optional[str]
    duration: float


class ASRModule:
    """Thin wrapper around Whisper for easier dependency injection and testing."""

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
        suppress_silence: bool = True,
        **load_kwargs,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.suppress_silence = suppress_silence
        self.load_kwargs = load_kwargs
        self._model = None
        logger.debug("ASRModule configured model=%s device=%s", model_name, device)

    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - depends on hardware
            return "cuda"
        if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - easier to debug missing deps
            raise RuntimeError(
                "Whisper is not installed. Install with `pip install -U openai-whisper torch`"
            ) from exc

        device = self._resolve_device()
        logger.info("Loading Whisper model %s on %s", self.model_name, device)
        self._model = whisper.load_model(self.model_name, device=device, **self.load_kwargs)

    def transcribe(self, audio_path: str | Path, **kwargs) -> ASRResult:
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        if self._model is None:
            self.load()

        start_ts = time.perf_counter()
        assert self._model is not None  # mypy friendliness
        logger.debug("Starting transcription for %s", audio.name)
        result = self._model.transcribe(str(audio), **kwargs)
        latency = time.perf_counter() - start_ts

        segments = [
            ASRSegment(text=seg["text"].strip(), start=float(seg["start"]), end=float(seg["end"]))
            for seg in result.get("segments", [])
        ]
        text = result.get("text", "").strip()
        language = result.get("language")

        logger.info("Transcribed %s in %.2fs", audio.name, latency)
        return ASRResult(text=text, segments=segments, language=language, duration=latency)
