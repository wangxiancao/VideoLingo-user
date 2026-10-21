from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Transcription:
    language: str
    segments: tuple[TranscriptSegment, ...]


class AsrBackend(Protocol):
    def transcribe(self, audio_path: str | Path, language: str | None = None) -> Transcription:
        ...