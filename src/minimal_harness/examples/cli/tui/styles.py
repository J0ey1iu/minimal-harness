CSS = """
$accent: #58a6ff;
$accent-dim: #388bfd;
$surface: #0d1117;
$surface-raised: #161b22;
$surface-overlay: #1c2128;
$border: #30363d;
$border-focus: #58a6ff;
$text-primary: #e6edf3;
$text-secondary: #8b949e;
$text-muted: #6e7681;
$user-accent: #3b82f6;
$user-bg: #1a2332;
$assistant-bg: #131820;
$tool-accent: #d29922;
$tool-bg: #1a1c20;
$result-accent: #3fb950;
$error-accent: #f85149;
$thinking-accent: #8957e5;

Screen {
    background: $surface;
    color: $text-primary;
}

Header {
    background: $surface-raised;
    color: $text-primary;
    dock: top;
    height: 1;
}

Footer {
    background: $surface-raised;
    color: $text-secondary;
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
    scrollbar-color: $border;
    scrollbar-color-hover: $text-muted;
    scrollbar-color-active: $accent-dim;
}

#bottom-bar {
    height: auto;
    margin: 0 1;
    padding: 0 0 1 0;
}

#model-bar {
    height: auto;
    padding: 1 1 0 1;
    background: $surface-overlay;
    border: round $border;
    margin: 0 0 1 0;
    display: none;
}

#model-bar Label {
    width: auto;
    padding: 0 1 0 0;
    color: $text-secondary;
    text-style: bold;
}

#model-input {
    width: 1fr;
    background: $surface;
    color: $text-primary;
    border: tall $border;
}

#model-input:focus {
    border: tall $border-focus;
}

#input-wrapper {
    height: auto;
    background: $surface-raised;
    border: round $border;
    padding: 0;
}

#input-wrapper:focus-within {
    border: round $accent;
}

#input {
    width: 1fr;
    background: transparent;
    color: $text-primary;
    border: none;
    padding: 1 2;
}

#input:focus {
    border: none;
}

#status-bar {
    height: 1;
    padding: 0 2;
    color: $text-muted;
}

#status-left {
    width: 1fr;
    color: $text-muted;
}

#status-right {
    width: auto;
    color: $text-muted;
}

/* ── Welcome ────────────────────────────────────────── */

.welcome {
    width: 100%;
    content-align: center middle;
    text-align: center;
    padding: 2 4;
    margin: 1 0;
    color: $text-secondary;
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
    color: $text-muted;
    padding: 0;
}

/* ── Messages ───────────────────────────────────────── */

.user-message {
    background: $user-bg;
    color: $text-primary;
    padding: 1 2;
    margin: 1 0 0 0;
    border-left: thick $user-accent;
}

.assistant-message {
    background: $assistant-bg;
    color: $text-primary;
    padding: 1 2;
    margin: 1 0 0 0;
    border-left: thick $accent-dim;
}

/* ── Tool call / result ─────────────────────────────── */

.tool-call {
    background: $tool-bg;
    color: $tool-accent;
    padding: 1 2;
    border-left: thick $tool-accent;
}

.tool-result {
    background: $tool-bg;
    color: $result-accent;
    padding: 1 2;
    border-left: thick $result-accent;
}

/* ── Thinking ───────────────────────────────────────── */

.thinking {
    color: $text-muted;
    text-style: italic;
}

/* ── System / model-change notice ───────────────────── */

.system-notice {
    width: 100%;
    text-align: center;
    color: $text-muted;
    padding: 0 2;
    margin: 1 0;
}

/* ── Role labels inside messages ────────────────────── */

.role-user {
    color: $user-accent;
    text-style: bold;
}

.role-assistant {
    color: $accent-dim;
    text-style: bold;
}
"""
