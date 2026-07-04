"""Transcript bundle definition and I/O helpers for the replay engine.

Transcripts are deterministic JSON records of LLM interactions. They are
used to completely isolate the test suite from the network during
integration testing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptBundle:
    """A serialized record of a single LLM interaction.

    Attributes:
        request:        The exact prompt string sent to the model.
        raw_response:   The raw text returned by the model.
        response_hash:  SHA-256 hex digest of the raw_response text.
        model:          The ModelId used.
        prompt_version: The version string of the prompt template.
        agent:          The name of the agent making the call.
    """

    request: str
    raw_response: str
    response_hash: str
    model: str
    prompt_version: str
    agent: str

    def save(self, directory: Path) -> Path:
        """Serialize and save to the given directory.

        The file is named using the last 16 characters of the response_hash
        as required by the spec.
        """
        directory.mkdir(parents=True, exist_ok=True)
        import hashlib
        req_hash = hashlib.sha256(self.request.encode("utf-8")).hexdigest()
        filename = f"{req_hash[-8:]}_{self.response_hash[-8:]}.json"
        path = directory / filename

        data = asdict(self)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path) -> TranscriptBundle:
        """Load a TranscriptBundle from a file path."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
