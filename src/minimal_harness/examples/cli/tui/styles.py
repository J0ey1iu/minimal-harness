CSS = """
Screen {
    background: $background;
    color: $text;
}

Header {
    background: $surface;
    color: $text;
    dock: top;
    height: 1;
}

Footer {
    background: $surface;
    color: $text-muted;
    dock: bottom;
}

#app-grid {
    layout: grid;
    grid-size: 1;
    grid-rows: 1fr auto;
    height: 1fr;
}

#chat-container {
    height: 1fr;
    margin: 0 1;
}

#history {
    height: 1fr;
    padding: 1 1;
    scrollbar-size: 1 1;
    scrollbar-color: $scrollbar;
    scrollbar-color-hover: $text-disabled;
    scrollbar-color-active: $accent;
}

#bottom-bar {
    height: auto;
    margin: 0 1;
    padding: 0 0 1 0;
}

#model-bar {
    height: auto;
    padding: 1 1 0 1;
    background: $panel;
    border: round $border;
    margin: 0 0 1 0;
    display: none;
}

#model-bar Label {
    width: auto;
    padding: 0 1 0 0;
    color: $text-muted;
    text-style: bold;
}

#model-input {
    width: 1fr;
    background: $background;
    color: $text;
    border: tall $border;
}

#model-input:focus {
    border: tall $accent;
}

#input-wrapper {
    height: auto;
    background: $surface;
    border: round $border;
    padding: 0;
}

#input-wrapper:focus-within {
    border: round $accent;
}

#input {
    width: 1fr;
    background: transparent;
    color: $text;
    border: none;
    padding: 1 2;
}

#input:focus {
    border: none;
}

#status-bar {
    height: 1;
    padding: 0 2;
    color: $text-disabled;
}

#status-left {
    width: 1fr;
    color: $text-disabled;
}

#status-right {
    width: auto;
    color: $text-disabled;
}

/* ── Welcome ────────────────────────────────────────── */

.welcome {
    width: 100%;
    content-align: center middle;
    text-align: center;
    padding: 2 4;
    margin: 1 0;
    color: $text-muted;
}

.welcome-title {
    width: 100%;
    text-align: center;
    color: $accent;
    text-style: bold;
    padding: 0;
}

.welcome-subtitle {
    width: 100%;
    text-align: center;
    color: $text-disabled;
    padding: 0;
}

/* ── Messages ───────────────────────────────────────── */

.user-message {
    background: $primary-background;
    color: $text;
    padding: 1 2;
    margin: 1 0 0 0;
    border-left: thick $primary;
}

.assistant-message {
    background: $secondary-background;
    color: $text;
    padding: 1 2;
    margin: 1 0 0 0;
    border-left: thick $accent;
}

/* ── Tool call / result ─────────────────────────────── */

.tool-call {
    background: $warning 8%;
    color: $warning;
    padding: 1 2;
    border-left: thick $warning;
}

.tool-result {
    background: $success 8%;
    color: $success;
    padding: 1 2;
    border-left: thick $success;
}

/* ── Thinking ───────────────────────────────────────── */

.thinking {
    color: $text-disabled;
    text-style: italic;
}

/* ── System / model-change notice ───────────────────── */

.system-notice {
    width: 100%;
    text-align: center;
    color: $text-disabled;
    padding: 0 2;
    margin: 1 0;
}

/* ── Role labels inside messages ────────────────────── */

.role-user {
    color: $primary;
    text-style: bold;
}

.role-assistant {
    color: $accent;
    text-style: bold;
}

/* ── System Prompt Modal ───────────────────────────────── */

SystemPromptScreen {
    background: $secondary-background;
    align: center middle;
}

#prompt-modal {
    background: $surface;
    border: thick $border;
    padding: 2;
    width: 80;
    height: 30;
}

.modal-title {
    width: 100%;
    text-align: center;
    color: $accent;
    text-style: bold;
    margin: 0 0 1 0;
}

#prompt-editor {
    width: 100%;
    height: 1fr;
    border: tall $border;
    margin: 1 0;
}

#prompt-editor:focus {
    border: tall $accent;
}

#modal-buttons {
    width: 100%;
    height: auto;
    padding: 1 0 0 0;
}

#save-hint {
    width: 1fr;
    background: transparent;
    color: $text-disabled;
    border: none;
    text-style: italic;
}

.modal-hint {
    width: auto;
    color: $text-disabled;
    text-style: italic;
}
"""
