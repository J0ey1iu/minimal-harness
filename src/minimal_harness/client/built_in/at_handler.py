"""@ command handling for TUI — file/directory picker."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from textual.widgets import Label, ListItem, ListView

if TYPE_CHECKING:
    from textual.timer import Timer

    from minimal_harness.client.built_in.widgets import ChatInput


class AtCommandHandler:
    def __init__(
        self,
        suggestion_list: ListView,
        input_widget: ChatInput,
        get_input_text: Callable[[], str],
        set_input_text: Callable[[str], None],
    ) -> None:
        self._suggestion_list = suggestion_list
        self._input = input_widget
        self._get_input_text = get_input_text
        self._set_input_text = set_input_text
        self._cwd = str(Path.cwd())
        self._entries: list[str] = []
        self._debounce_timer: Timer | None = None
        self._filter_seq: int = 0
        # Pre-cached file list (populated from git ls-files)
        self._file_cache: list[str] | None = None
        self._file_cache_lower: list[str] | None = None
        self._is_git_repo: bool | None = None
        asyncio.ensure_future(self._build_cache())

    async def _build_cache(self) -> None:
        """Try git ls-files to build an instant file list from the index."""
        try:
            files = await asyncio.to_thread(self._git_ls_files, self._cwd)
            if files is not None:
                self._file_cache = files
                self._file_cache_lower = [f.lower() for f in files]
                self._is_git_repo = True
            else:
                self._is_git_repo = False
        except Exception:
            self._is_git_repo = False

    @staticmethod
    def _git_ls_files(cwd: str) -> list[str] | None:
        """Run git ls-files. Returns list of relative paths or None if not a git repo."""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            lines = result.stdout.splitlines()
            return lines if lines else None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _filter_cached(self, filter_text: str) -> list[str]:
        """Fast in-memory filter over pre-cached git file list."""
        cache = self._file_cache
        cache_lower = self._file_cache_lower
        if not cache or not cache_lower:
            return []
        if not filter_text:
            return cache[:10]
        lower = filter_text.lower()
        results: list[str] = []
        for i, fl in enumerate(cache_lower):
            if lower in fl:
                results.append(cache[i])
                if len(results) >= 10:
                    break
        return results

    @staticmethod
    def _rglob_fallback(cwd: str, filter_text: str) -> list[str]:
        """Fallback for non-git repos: rglob with glob pattern to push matching to C."""
        if not filter_text:
            results: list[str] = []
            try:
                for p in Path(cwd).rglob("*"):
                    try:
                        rel = str(p.relative_to(cwd))
                    except ValueError:
                        continue
                    results.append(rel)
                    if len(results) >= 10:
                        break
            except (PermissionError, OSError):
                pass
            return results
        # Use glob pattern so fnmatch filters in C, not Python
        pattern = f"*{filter_text}*"
        results: list[str] = []
        try:
            for p in Path(cwd).rglob(pattern):
                try:
                    rel = str(p.relative_to(cwd))
                except ValueError:
                    continue
                results.append(rel)
                if len(results) >= 10:
                    break
        except (PermissionError, OSError):
            pass
        return results

    def _show_suggestions(self, entries: list[str]) -> None:
        if not entries:
            self._hide_suggestions()
            return
        self._entries = entries
        self._suggestion_list.clear()
        for rel in entries:
            self._suggestion_list.append(ListItem(Label(rel)))
        self._suggestion_list.add_class("visible")
        self._input.set_at_active(True)
        if self._suggestion_list.children:
            self._suggestion_list.index = 0

    def _cancel_debounce(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
            self._debounce_timer = None

    def _hide_suggestions(self) -> None:
        self._cancel_debounce()
        self._suggestion_list.remove_class("visible")
        self._suggestion_list.clear()
        self._entries = []
        self._input.set_at_active(False)

    def _insert_path(self, rel: str) -> None:
        text = self._get_input_text()
        at_pos = text.rfind("@")
        if at_pos == -1:
            return
        abs_path = str(Path(self._cwd) / rel)
        start = self._offset_to_location(at_pos)
        end = self._offset_to_location(len(text))
        self._input.replace(abs_path, start, end)

    def _offset_to_location(self, offset: int) -> tuple[int, int]:
        before = self._get_input_text()[:offset]
        lines = before.split("\n")
        return (len(lines) - 1, len(lines[-1]))

    def on_at_command_show(self, text: str) -> None:
        self._cancel_debounce()
        self._filter_seq += 1
        seq = self._filter_seq
        self._debounce_timer = self._input.set_timer(
            0.08, lambda t=text, s=seq: asyncio.ensure_future(self._do_show(t, s))
        )

    async def _do_show(self, text: str, seq: int) -> None:
        self._debounce_timer = None
        if seq != self._filter_seq:
            return
        idx = text.rfind("@")
        if idx == -1:
            self._hide_suggestions()
            return
        filter_text = text[idx + 1 :]
        # Use git cache if available (instant), otherwise rglob fallback in thread
        if self._is_git_repo and self._file_cache:
            filtered = self._filter_cached(filter_text)
        else:
            filtered = await asyncio.to_thread(
                self._rglob_fallback, self._cwd, filter_text
            )
        if seq != self._filter_seq:
            return
        self._show_suggestions(filtered)

    def on_at_command_hide(self) -> None:
        self._hide_suggestions()

    def on_at_command_navigate_up(self) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_up()

    def on_at_command_navigate_down(self) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_down()

    def on_at_command_select(self) -> None:
        sl = self._suggestion_list
        if not sl.children or sl.index is None:
            return
        idx = sl.index
        if 0 <= idx < len(self._entries):
            self._insert_path(self._entries[idx])
            self._hide_suggestions()

    def on_list_view_selected(self, idx: int | None) -> None:
        if not self._suggestion_list.has_class("visible"):
            return
        if idx is None:
            return
        if 0 <= idx < len(self._entries):
            self._insert_path(self._entries[idx])
            self._hide_suggestions()
