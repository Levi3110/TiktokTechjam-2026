#!/usr/bin/env python3
"""Send one simple chat-completion request to an OpenAI-compatible Qwen endpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # The API key can still be supplied through the environment.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen3.6-27b"
ENV_PATH = PROJECT_ROOT / "techjam-conversational-search-main" / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="User message to send to Qwen")
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System message (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL),
        help=f"Model identifier (default: QWEN_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL"),
        help="OpenAI-compatible endpoint (required via QWEN_BASE_URL or this option)",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Set QWEN_BASE_URL in .env or pass --base-url")
    return args


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=False)
    args = parse_args()
    # vLLM accepts unauthenticated requests unless it was started with --api-key.
    # The OpenAI client still requires a non-empty value locally.
    api_key = (
        os.environ.get("QWEN_API_KEY")
        or os.environ.get("CHENGQI_QWEN_API_KEY")
        or "not-needed-for-local-vllm"
    )

    try:
        from openai import OpenAI
    except ImportError:
        print("Install the OpenAI client first: python -m pip install openai", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body={"top_k": -1, "chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content
    if not content:
        print("Qwen returned an empty response.", file=sys.stderr)
        return 1
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
