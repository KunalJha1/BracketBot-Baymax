"""Deterministic voice intents with an optional OpenRouter fallback.

Physical actions are selected only from the phrase table in this module.  The
language model receives conversational turns, but it is never allowed to
choose or synthesize a robot action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
import re
import threading
from typing import Callable
from urllib import error, request


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SYSTEM_PROMPT = (
    "You are Baymax, a warm embodied home robot assistant. Answer in one or "
    "two short, natural sentences because your response will be spoken aloud. "
    "Do not claim to be a medical professional, diagnose, or prescribe. "
    "Robot motions are handled by a separate deterministic safety system; "
    "never say that you performed a physical action."
)


class RouteKind(str, Enum):
    ACTION = "action"
    QUESTION = "question"
    CHAT = "chat"
    EMPTY = "empty"
    ERROR = "error"


@dataclass(frozen=True)
class RouteDecision:
    kind: RouteKind
    utterance: str
    action: str | None = None
    reply: str | None = None


# Keep this deliberately explicit. Adding an alias here is what grants spoken
# language permission to start that motion.
ACTION_ALIASES = {
    "hug": frozenset(
        {
            "hug",
            "hug me",
            "give me a hug",
            "can i have a hug",
            "could i have a hug",
            "may i have a hug",
            "i need a hug",
        }
    ),
    "fist bump": frozenset(
        {
            "fist bump",
            "fist bump me",
            "give me a fist bump",
            "can i get a fist bump",
            "can i have a fist bump",
        }
    ),
    "handshake": frozenset(
        {
            "handshake",
            "shake my hand",
            "give me a handshake",
            "can i have a handshake",
        }
    ),
    "wave": frozenset(
        {
            "wave",
            "wave at me",
            "give me a wave",
            "can you wave",
            "say hello",
            "greet me",
        }
    ),
}

_WAKE_PREFIX = re.compile(r"^(?:(?:hey|hi|hello|ok|okay)\s+)?(?:baymax|bamax)\s+")
_LEADING_POLITE = re.compile(r"^(?:please\s+|can you please\s+|could you please\s+)")
_TRAILING_POLITE = re.compile(r"\s+(?:please|for me|right now|now)$")
_QUESTION_OPENERS = frozenset(
    {
        "am", "are", "can", "could", "did", "do", "does", "explain",
        "how", "is", "may", "should", "tell", "what", "when", "where",
        "which", "who", "why", "will", "would",
    }
)


def normalize_utterance(text: str) -> str:
    """Normalize speech-to-text output without fuzzy action matching."""
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"[^a-z0-9'\s-]", " ", normalized)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = _WAKE_PREFIX.sub("", normalized, count=1)
    normalized = _LEADING_POLITE.sub("", normalized, count=1)
    normalized = _TRAILING_POLITE.sub("", normalized, count=1)
    return normalized.strip()


def match_action(text: str) -> str | None:
    normalized = normalize_utterance(text)
    for action, aliases in ACTION_ALIASES.items():
        if normalized in aliases:
            return action
    return None


def utterances_match(transcript: str, claimed: str) -> bool:
    """Verify a model tool argument against finalized speech transcription."""
    normalized_transcript = normalize_utterance(transcript)
    return bool(normalized_transcript) and normalized_transcript == normalize_utterance(claimed)


def is_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    normalized = normalize_utterance(stripped)
    first_word = normalized.split(" ", 1)[0] if normalized else ""
    return first_word in _QUESTION_OPENERS


class OpenRouterError(RuntimeError):
    """A user-safe wrapper for OpenRouter request or response failures."""


class OpenRouterClient:
    """Small dependency-free OpenRouter chat client with bounded history."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = 20.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_messages: int = 8,
        opener: Callable[..., object] = request.urlopen,
    ):
        self.api_key = (
            os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        )
        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.max_history_messages = max(0, max_history_messages)
        self._opener = opener
        self._history: list[dict[str, str]] = []
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def ask(self, utterance: str) -> str:
        if not self.api_key:
            raise OpenRouterError(
                "OpenRouter is not configured yet. Set OPENROUTER_API_KEY to enable questions."
            )

        with self._lock:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self._history,
                {"role": "user", "content": utterance},
            ]
            payload = json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 180,
                    "temperature": 0.4,
                }
            ).encode("utf-8")
            api_request = request.Request(
                OPENROUTER_URL,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/BracketBot/BracketBot",
                    "X-OpenRouter-Title": "BracketBot Baymax",
                },
            )

            try:
                with self._opener(api_request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                raise OpenRouterError(f"OpenRouter returned HTTP {exc.code}: {detail}") from exc
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

            try:
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                reply = str(content).strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise OpenRouterError("OpenRouter returned an unexpected response.") from exc
            if not reply:
                raise OpenRouterError("OpenRouter returned an empty response.")

            self._history.extend(
                [
                    {"role": "user", "content": utterance},
                    {"role": "assistant", "content": reply},
                ]
            )
            if self.max_history_messages:
                self._history = self._history[-self.max_history_messages :]
            else:
                self._history.clear()
            return reply


class VoiceRouter:
    """Route explicit action phrases locally and all other speech to chat."""

    def __init__(self, llm: OpenRouterClient):
        self.llm = llm

    def route(self, utterance: str) -> RouteDecision:
        utterance = utterance.strip()
        if not utterance:
            return RouteDecision(RouteKind.EMPTY, utterance)

        action = match_action(utterance)
        if action:
            return RouteDecision(
                RouteKind.ACTION,
                utterance,
                action=action,
                reply=f"Of course. Starting the {action} now.",
            )

        kind = RouteKind.QUESTION if is_question(utterance) else RouteKind.CHAT
        try:
            reply = self.llm.ask(utterance)
        except OpenRouterError as exc:
            return RouteDecision(RouteKind.ERROR, utterance, reply=str(exc))
        return RouteDecision(kind, utterance, reply=reply)
