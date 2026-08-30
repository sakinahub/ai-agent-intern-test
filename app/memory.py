from __future__ import annotations


class ConversationMemory:
    """
    Lightweight conversation memory for a single session.
    """

    def __init__(self, max_messages: int = 12):
        self.max_messages = max_messages
        self.messages: list[dict] = []

    def add_message(
        self,
        role: str,
        content: str,
    ):
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        # Keep memory bounded.
        self.messages = self.messages[
            -self.max_messages:
        ]

    def get_messages(self) -> list[dict]:
        return list(self.messages)

    def get_context(self) -> str:
        if not self.messages:
            return ""

        lines = []

        for message in self.messages:
            role = message["role"].upper()

            lines.append(
                f"{role}: {message['content']}"
            )

        return "\n".join(lines)

    def clear(self):
        self.messages.clear()