from __future__ import annotations

from rich.console import Console

from minimal_harness.client.built_in.markdown_styles import (
    LazyMarkdown,
    resolve_code_theme,
)


class TestResolveCodeTheme:
    def test_dark_theme_returns_native(self):
        dark_themes = [
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
        ]
        for theme in dark_themes:
            assert resolve_code_theme(theme) == "native", f"{theme} should be native"

    def test_light_theme_returns_fruity(self):
        light_themes = [
            "textual-light",
            "catppuccin-latte",
            "solarized-light",
            "atom-one-light",
            "rose-pine-dawn",
        ]
        for theme in light_themes:
            assert resolve_code_theme(theme) == "fruity", f"{theme} should be fruity"

    def test_unknown_theme_defaults_to_fruity(self):
        assert resolve_code_theme("nonexistent-theme") == "fruity"


class TestLazyMarkdown:
    def test_create(self):
        md = LazyMarkdown("# Hello", code_theme="native")
        assert md.text == "# Hello"
        assert md.code_theme == "native"
        assert md._md is None

    def test_create_default_code_theme(self):
        md = LazyMarkdown("hello")
        assert md.code_theme is None
        assert md._md is None

    def test_lazy_initialization_on_render(self):
        md = LazyMarkdown("**bold**", code_theme="native")
        assert md._md is None
        from rich.console import Console

        console = Console(width=80)
        segments = list(console.render(md))
        assert md._md is not None
        assert len(segments) > 0

    def test_render_caches_markdown(self):
        md = LazyMarkdown("test", code_theme="native")
        console = Console(width=80)
        list(console.render(md))
        first_md = md._md
        list(console.render(md))
        assert md._md is first_md
