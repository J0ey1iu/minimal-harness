from textual.widgets import Static


class ChatMessage(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_focus = False


class ToolCallWidget(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_focus = False


class ToolResultWidget(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_focus = False


class ThinkingWidget(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_focus = False


class MemoryStatus(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_focus = False
