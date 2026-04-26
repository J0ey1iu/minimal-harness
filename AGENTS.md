# Agent coding guide

## Must do

1. When you need to start the project in any way, use python interpreter in ./.venv/bin/python.
2. Run `uv run ruff check --fix`, `uv run ruff check --select I --fix` and `uv run pyright .` after finished code editing. And fix the errors and warning found by these wonderful linters.
3. Run `uv run ruff format` to format the file for standard convention formatting.

## Must NOT do

1. Don't ever commit anything without a user asking you to.
