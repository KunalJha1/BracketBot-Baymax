#!/usr/bin/env python3
"""Benchmark OpenRouter models for the spoken assistant.

Measures what a person hears: time to the first complete sentence (when TTS
could start) and time to the full reply, streamed, from the robot's network.

    scripts/bot push scripts/bench_voice_llm.py --to /tmp
    scripts/bot 'set -a; . ~/bbapps/.env; python3 /tmp/bench_voice_llm.py'
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from urllib import request

URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = (
    "You are BracketBot, a warm embodied home robot assistant. Answer in one "
    "or two short, natural sentences because your response will be spoken aloud."
)
QUESTIONS = [
    "why is the sky blue",
    "how many legs does a spider have",
    "tell me a quick joke",
]
# (label, model, extra payload)
CONFIGS = [
    ("gpt-oss-20b default/throughput", "openai/gpt-oss-20b", {"provider": {"sort": "throughput"}}),
    ("gpt-oss-20b low/latency", "openai/gpt-oss-20b", {"provider": {"sort": "latency"}, "reasoning": {"effort": "low"}}),
    ("gpt-oss-120b low/latency", "openai/gpt-oss-120b", {"provider": {"sort": "latency"}, "reasoning": {"effort": "low"}}),
    ("gemini-2.5-flash-lite", "google/gemini-2.5-flash-lite", {"provider": {"sort": "latency"}}),
    ("gemini-2.5-flash no-think", "google/gemini-2.5-flash", {"provider": {"sort": "latency"}, "reasoning": {"enabled": False}}),
    ("gpt-4.1-nano", "openai/gpt-4.1-nano", {"provider": {"sort": "latency"}}),
    ("gpt-4.1-mini", "openai/gpt-4.1-mini", {"provider": {"sort": "latency"}}),
    ("claude-haiku-4.5", "anthropic/claude-haiku-4.5", {"provider": {"sort": "latency"}}),
    ("llama-3.3-70b", "meta-llama/llama-3.3-70b-instruct", {"provider": {"sort": "latency"}}),
    ("llama-4-scout", "meta-llama/llama-4-scout", {"provider": {"sort": "latency"}}),
    ("mistral-small-3.2", "mistralai/mistral-small-3.2-24b-instruct", {"provider": {"sort": "latency"}}),
    ("qwen3-32b no-think", "qwen/qwen3-32b", {"provider": {"sort": "latency"}, "reasoning": {"enabled": False}}),
]
SENTENCE_END = re.compile(r"[.!?](\s|$)")


def one(model, extra, question, key):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        "max_tokens": 180,
        "temperature": 0.4,
        "stream": True,
        **extra,
    }
    req = request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    start = time.monotonic()
    text, first_token, first_sentence, provider = "", None, None, ""
    with request.urlopen(req, timeout=30) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            provider = event.get("provider", provider)
            choices = event.get("choices") or [{}]
            piece = (choices[0].get("delta") or {}).get("content") or ""
            if piece:
                now = time.monotonic() - start
                first_token = first_token if first_token is not None else now
                text += piece
                if first_sentence is None and len(text) >= 24 and SENTENCE_END.search(text):
                    first_sentence = now
    total = time.monotonic() - start
    return first_token or total, first_sentence or total, total, provider, text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="substring filter on config label")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    print(f"{'config':34} {'ttft':>6} {'1st-sent':>8} {'total':>6} {'worst':>6}  providers")
    for label, model, extra in CONFIGS:
        if args.only and args.only not in label:
            continue
        rows, providers, sample, failures = [], set(), "", 0
        for _ in range(args.rounds):
            for question in QUESTIONS:
                try:
                    ttft, sentence, total, provider, text = one(model, extra, question, key)
                except Exception as exc:  # noqa: BLE001 - a benchmark reports, not raises
                    failures += 1
                    sample = sample or f"ERROR {type(exc).__name__}: {exc}"
                    continue
                rows.append((ttft, sentence, total))
                providers.add(provider)
                sample = text if not sample or sample.startswith("ERROR") else sample
        if not rows:
            print(f"{label:34} FAILED  {sample[:90]}")
            continue
        med = [statistics.median(col) for col in zip(*rows)]
        worst = max(row[2] for row in rows)
        print(f"{label:34} {med[0]:6.2f} {med[1]:8.2f} {med[2]:6.2f} {worst:6.2f}  "
              f"{','.join(sorted(providers))} fail={failures}", flush=True)
        print(f"    e.g. {sample[:110]!r}", flush=True)


if __name__ == "__main__":
    main()
