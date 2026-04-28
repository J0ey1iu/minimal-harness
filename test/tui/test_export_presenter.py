from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from minimal_harness.client.built_in.display import ExportEntry
from minimal_harness.client.built_in.export_presenter import ExportPresenter


def _make_export_history() -> list[ExportEntry]:
    return []


class TestExportPresenter:
    def test_export_svg_success(self, tmp_path: Path):
        get_theme = MagicMock(return_value="nord")
        say = MagicMock()
        presenter = ExportPresenter(get_theme=get_theme, say=say)

        export_history = [
            ExportEntry(text="Hello"),
            ExportEntry(text="**bold**", is_markdown=True),
        ]
        output_path = str(tmp_path / "chat.svg")
        presenter.export_svg(output_path, export_history, chat_width=80)

        assert Path(output_path).exists()
        svg_content = Path(output_path).read_text(encoding="utf-8")
        assert "<svg" in svg_content or "xml" in svg_content.lower()
        say.assert_called_once()

    def test_export_svg_with_styled_text(self, tmp_path: Path):
        get_theme = MagicMock(return_value="nord")
        say = MagicMock()
        presenter = ExportPresenter(get_theme=get_theme, say=say)

        export_history = [
            ExportEntry(text="Error occurred", style="bold red"),
        ]
        output_path = str(tmp_path / "styled.svg")
        presenter.export_svg(output_path, export_history, chat_width=80)

        assert Path(output_path).exists()

    def test_export_svg_handles_error(self, tmp_path: Path):
        get_theme = MagicMock(return_value="nord")
        say = MagicMock()
        presenter = ExportPresenter(get_theme=get_theme, say=say)

        export_history = [
            ExportEntry(text="test"),
        ]
        output_path = str(tmp_path / "nonexistent" / "subdir" / "chat.svg")
        presenter.export_svg(output_path, export_history, chat_width=80)

        assert Path(output_path).exists()

    def test_export_svg_height_calculation(self, tmp_path: Path):
        get_theme = MagicMock(return_value="nord")
        say = MagicMock()
        presenter = ExportPresenter(get_theme=get_theme, say=say)

        export_history = [
            ExportEntry(text="line1\nline2\nline3"),
        ]
        output_path = str(tmp_path / "height_test.svg")
        presenter.export_svg(output_path, export_history, chat_width=80)

        assert Path(output_path).exists()

    def test_export_svg_with_markdown_height_calculation(self, tmp_path: Path):
        get_theme = MagicMock(return_value="nord")
        say = MagicMock()
        presenter = ExportPresenter(get_theme=get_theme, say=say)

        long_text = "word " * 100
        export_history = [
            ExportEntry(text=long_text, is_markdown=True),
        ]
        output_path = str(tmp_path / "md_height.svg")
        presenter.export_svg(output_path, export_history, chat_width=40)

        assert Path(output_path).exists()

    def test_empty_export_history(self, tmp_path: Path):
        get_theme = MagicMock(return_value="nord")
        say = MagicMock()
        presenter = ExportPresenter(get_theme=get_theme, say=say)

        output_path = str(tmp_path / "empty.svg")
        presenter.export_svg(output_path, [], chat_width=80)

        assert Path(output_path).exists()
