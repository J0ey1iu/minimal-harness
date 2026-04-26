"""Custom Markdown classes with styled tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, ConsoleOptions
from rich.markdown import Markdown as BaseMarkdown
from rich.markdown import MarkdownContext, MarkdownElement
from rich.table import Table

if TYPE_CHECKING:
    from rich.markdown import TableBodyElement, TableHeaderElement


class BorderedTableElement(MarkdownElement):
    """Table element with visible borders and lines."""

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
            box=box.SQUARE,
            pad_edge=False,
            style="markdown.table.border",
            show_edge=True,
            show_lines=True,
            collapse_padding=True,
        )

        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("markdown.table.header")
                table.add_column(heading)

        if self.body is not None:
            for row in self.body.rows:
                row_content = [element.content for element in row.cells]
                table.add_row(*row_content)

        yield table


class BorderedMarkdown(BaseMarkdown):
    """Markdown renderer with bordered tables."""

    elements = BaseMarkdown.elements.copy()
    elements["table_open"] = BorderedTableElement
