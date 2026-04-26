"""Export/sharing logic extracted from TUIApp."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from minimal_harness.client.built_in.markdown_styles import (
    LazyMarkdown,
    resolve_code_theme,
)

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


class ExportPresenter:
    def __init__(self, app: TUIApp) -> None:
        self._app = app

    def export_svg(self, path: str) -> None:
        width = self._app._chat_width or 80
        height = 24
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
                for text, style, is_md in self._app._export_history:
                    if is_md:
                        code_theme = resolve_code_theme(self._app.theme)
                        console.print(LazyMarkdown(text, code_theme=code_theme))
                    elif style:
                        console.print(Text(text, style=style))
                    else:
                        console.print(Text(text))
            svg = console.export_svg(title="Minimal Harness Chat")
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(svg, encoding="utf-8")
            self._app.say(f"✓ Chat exported → {path}", "bold bright_green")
        except Exception as e:
            self._app.say(f"✗ {e}", "bold bright_red")