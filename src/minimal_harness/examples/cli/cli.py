import argparse
import os

from .tui import ChatTUI


def main():
    parser = argparse.ArgumentParser(
        description="minimal-harness CLI - Chat with LLMs via OpenAI-compatible API",
        prog="mh",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("MH_BASE_URL"),
        help="API base URL (required if not set in LITELLM_BASE_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("MH_API_KEY") or os.environ.get("API_KEY"),
        help="API key (falls back to OPENAI_API_KEY or API_KEY env vars)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MH_MODEL", "minimax-m2.1"),
        help="Model name (default: minimax-m2.1, or MH_MODEL env var)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the assistant",
    )

    args = parser.parse_args()

    if not args.base_url:
        parser.error("--base-url is required (or set MH_BASE_URL env var)")

    app = ChatTUI(
        base_url=args.base_url,
        api_key=args.api_key or "",
        model=args.model,
        system_prompt=args.system_prompt,
    )
    app.run()


if __name__ == "__main__":
    main()
