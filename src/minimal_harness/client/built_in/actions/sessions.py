"""Session selection action — handles /sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import SessionSelectScreen
from minimal_harness.client.events import to_client_event

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_sessions(app: TUIApp) -> None:
    if app._ctrl.streaming:
        return
    sessions = app._ctrl.get_all_sessions_metadata()

    def done(session_id: str | None) -> None:
        if not session_id or app._session_manager is None:
            return
        d = app._chat_display
        if d is None:
            return
        app._first = True

        session = app._ctrl.load_session_from_disk(session_id)
        if session:
            app._ctrl.switch_session(session_id)
            app._update_top_bar()
            success, inputs = app._session_manager.replay_session(
                session,
                clear_committed=app._clear_committed,
                clear_buf=app._ctrl.buf.clear,
            )
            if success:
                app._first = False
                app._banner_widget.display = False
                app._chat.display = True
                app._input.input_history = inputs
                app._input.reset_history_index()
                if session_id in app._ctrl._active_runs:
                    events, finished = app._ctrl.drain_session_events(session_id)
                    sess = app._ctrl.current_session
                    if events and sess and d:
                        for event in events:
                            d.handle_event(
                                to_client_event(event),
                                buf=app._ctrl.buf,
                                memory=sess.memory,
                            )
                            d.tick(app._ctrl.buf, True)
                    if not finished:
                        app._set_streaming(True)
                    else:
                        if not app._ctrl.buf.flushed:
                            d.flush(app._ctrl.buf)
                        app._ctrl.buf.clear()

    app.push_screen(SessionSelectScreen(sessions), done)
