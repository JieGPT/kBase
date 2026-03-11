from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime


class Message:
    def __init__(self, role: str, content: str, timestamp: datetime = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()


class ConversationBuffer:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.messages: List[Message] = []

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))
        self._trim_if_needed()

    def add_assistant_message(self, content: str):
        self.messages.append(Message(role="assistant", content=content))
        self._trim_if_needed()

    def _trim_if_needed(self):
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2) :]

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def clear(self):
        self.messages = []

    def get_context_string(self) -> str:
        if not self.messages:
            return ""
        lines = []
        for msg in self.messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
