"""Export/sharing logic extracted from TUIApp."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.text import Text

from minimal_harness.client.built_in.markdown_styles import (
    LazyMarkdown,
    resolve_code_theme,
)


class ExportPresenter:
    def __init__(
        self,
        get_theme: Callable[[], str],
        say: Callable[..., None],
    ) -> None:
        self._get_theme = get_theme
        self._say = say

    def export_svg(
        self,
        path: str,
        export_history: list[tuple[str, str | None, bool]],
        chat_width: int = 80,
    ) -> None:
        width = chat_width or 80
        lines = 0
        for text, _, is_md in export_history:
            if is_md:
                lines += text.count("\n") + max(1, len(text) // max(width, 1))
            else:
                lines += text.count("\n") + 1
        height = max(24, lines + 4)
        buf = StringIO()
        console = Console(
            file=buf,
            force_terminal=True,
            width=width,
            height=height,
            record=True,
            legacy_windows=False,
            color_system="truecolor",
        )
        try:
            with console:
                for text, style, is_md in export_history:
                    if is_md:
                        code_theme = resolve_code_theme(self._get_theme())
                        console.print(LazyMarkdown(text, code_theme=code_theme))
                    elif style:
                        console.print(Text(text, style=style))
                    else:
                        console.print(Text(text))
            svg = console.export_svg(title="Minimal Harness Chat")
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(svg, encoding="utf-8")
            self._say(f"\u2713 Chat exported \u2192 {path}", "bold bright_green")
        except Exception as e:
            self._say(f"\u2717 {e}", "bold bright_red")
