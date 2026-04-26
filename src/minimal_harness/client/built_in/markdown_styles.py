"""Custom Markdown renderer with refined terminal styling."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, ConsoleOptions
from rich.markdown import CodeBlock as BaseCodeBlock
from rich.markdown import Heading, MarkdownContext, MarkdownElement
from rich.markdown import Markdown as BaseMarkdown
from rich.measure import Measurement
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.markdown import TableBodyElement, TableHeaderElement

_DARK_THEMES = frozenset({
    "textual-dark",
    "tokyo-night",
    "catppuccin-mocha",
    "catppuccin-frappe",
    "catppuccin-macchiato",
    "rose-pine",
    "rose-pine-moon",
    "flexoki",
    "textual-ansi",
    "atom-one-dark",
    "nord",
    "gruvbox",
    "monokai",
    "dracula",
    "solarized-dark",
})


def resolve_code_theme(theme: str) -> str:
    return "native" if theme in _DARK_THEMES else "fruity"


class LeftHeading(Heading):
    """Heading with all levels left-aligned."""

    LEVEL_ALIGN = {
        "h1": "left",
        "h2": "left",
        "h3": "left",
        "h4": "left",
        "h5": "left",
        "h6": "left",
    }


class StyledTableElement(MarkdownElement):
    """Table with rounded borders, readable spacing, and emphasized headers."""

    def __init__(self) -> None:
        self.header: TableHeaderElement | None = None
        self.body: TableBodyElement | None = None

    def on_child_close(
        self, context: MarkdownContext, child: MarkdownElement
    ) -> bool:
        from rich.markdown import TableBodyElement, TableHeaderElement

        if isinstance(child, TableHeaderElement):
            self.header = child
        elif isinstance(child, TableBodyElement):
            self.body = child
        else:
            raise RuntimeError("Couldn't process markdown table.")
        return False

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ):
        table = Table(
            box=box.ROUNDED,
            pad_edge=False,
            style="markdown.table.border",
            show_edge=True,
            show_lines=False,
            collapse_padding=True,
            padding=(0, 2),
            header_style="bold",
        )

        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("bold markdown.table.header")
                table.add_column(heading)

        if self.body is not None:
            for row in self.body.rows:
                row_content = [element.content for element in row.cells]
                table.add_row(*row_content)

        yield table


class StyledCodeBlock(BaseCodeBlock):
    """Code block wrapped in a subtle rounded panel."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ):
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=(0, 1),
        )
        yield Panel(
            syntax,
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 0),
        )


class AppMarkdown(BaseMarkdown):
    """Markdown renderer with refined terminal styling for the TUI."""

    elements = BaseMarkdown.elements.copy()
    elements["table_open"] = StyledTableElement
    elements["fence"] = StyledCodeBlock
    elements["code_block"] = StyledCodeBlock
    elements["heading_open"] = LeftHeading

    def __init__(self, markup: str, code_theme: str | None = None, **kwargs):
        if code_theme is None:
            code_theme = "monokai"
        super().__init__(markup, code_theme=code_theme, **kwargs)


class LazyMarkdown:
    """Markdown renderable that re-renders at the display width for responsive layouts."""

    def __init__(self, text: str, code_theme: str | None = None) -> None:
        self.text = text
        self.code_theme = code_theme
        self._cache_width = 0
        self._cache_result: Text | None = None

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        width = max(options.max_width, 20)
        if width == self._cache_width and self._cache_result is not None:
            yield self._cache_result
            return

        buf = StringIO()
        with Console(file=buf, force_terminal=True, width=width) as c:
            c.print(AppMarkdown(self.text, code_theme=self.code_theme))
        result = Text.from_ansi(buf.getvalue())
        self._cache_width = width
        self._cache_result = result
        yield result

    def __rich_measure__(self, console: Console, options: ConsoleOptions):
        return Measurement(0, options.max_width)
