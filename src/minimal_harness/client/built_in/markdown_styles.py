"""Custom Markdown renderer with refined terminal styling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, ConsoleOptions
from rich.markdown import BlockQuote as BaseBlockQuote
from rich.markdown import CodeBlock as BaseCodeBlock
from rich.markdown import (
    Heading,
    MarkdownContext,
    MarkdownElement,
    TableBodyElement,
    TableHeaderElement,
)
from rich.markdown import Markdown as BaseMarkdown
from rich.measure import Measurement
from rich.panel import Panel
from rich.rule import Rule
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from rich.markdown import TableBodyElement, TableHeaderElement

_DARK_THEMES = frozenset(
    {
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
    }
)


def resolve_code_theme(theme: str) -> str:
    return "native" if theme in _DARK_THEMES else "fruity"


MD_THEME = Theme(
    {
        "markdown.link_url": "underline",
    }
)


class StyledHeading(Heading):
    """Heading with level-appropriate visual styling."""

    LEVEL_ALIGN = {
        "h1": "left",
        "h2": "left",
        "h3": "left",
        "h4": "left",
        "h5": "left",
        "h6": "left",
    }

    LEVEL_STYLES = {
        "h1": "bold",
        "h2": "bold",
        "h3": "bold italic",
        "h4": "italic",
        "h5": "dim",
        "h6": "dim italic",
    }

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        text = self.text.copy()
        text.justify = self.LEVEL_ALIGN.get(self.tag, "left")
        style = self.LEVEL_STYLES.get(self.tag, "")
        if style:
            text.stylize(style)
        yield text
        if self.tag == "h1":
            yield Rule(style="dim", characters="─")
            yield Text()


class StyledTableElement(MarkdownElement):
    """Table with rounded borders, readable spacing, and emphasized headers."""

    def __init__(self) -> None:
        self.header: TableHeaderElement | None = None
        self.body: TableBodyElement | None = None

    def on_child_close(self, context: MarkdownContext, child: MarkdownElement) -> bool:
        if isinstance(child, TableHeaderElement):
            self.header = child
        elif isinstance(child, TableBodyElement):
            self.body = child
        return False

    def __rich_console__(self, console: Console, options: ConsoleOptions):
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


class StyledBlockQuote(BaseBlockQuote):
    """Block quote with a clean vertical bar prefix."""

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        render_options = options.update(width=options.max_width - 4)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        style = self.style
        new_line = Segment("\n")
        padding = Segment("┃ ", style)
        yield Text()
        for line in lines:
            yield padding
            yield from line
            yield new_line
        yield Text()


class StyledHorizontalRule(MarkdownElement):
    """Horizontal rule as a thin dim line."""

    new_line = False

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        yield Rule(style="dim", characters="─")
        yield Text()


class StyledCodeBlock(BaseCodeBlock):
    """Code block wrapped in a subtle rounded panel."""

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=(1, 2),
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
    elements["heading_open"] = StyledHeading
    elements["blockquote_open"] = StyledBlockQuote
    elements["hr"] = StyledHorizontalRule

    def __init__(self, markup: str, code_theme: str | None = None, **kwargs):
        if code_theme is None:
            code_theme = "monokai"
        super().__init__(markup, code_theme=code_theme, **kwargs)


class LazyMarkdown:
    """Markdown renderable with width-responsive caching."""

    def __init__(self, text: str, code_theme: str | None = None) -> None:
        self.text = text
        self.code_theme = code_theme
        self._md: AppMarkdown | None = None

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        if self._md is None:
            self._md = AppMarkdown(self.text, code_theme=self.code_theme)
        yield self._md

    def __rich_measure__(self, console: Console, options: ConsoleOptions):
        return Measurement(20, options.max_width)
