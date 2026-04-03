from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class QASession:
    session_id: str
    file_id: str
    questions: list[str]
    targets: list[dict]
    answers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.answers:
            self.answers = [""] * len(self.questions)

    def record_answer(self, answer: str, index: int | None) -> None:
        if isinstance(index, int) and index >= 0:
            if index >= len(self.answers):
                self.answers.extend([""] * (index + 1 - len(self.answers)))
            self.answers[index] = answer
            return
        self.answers.append(answer)


class QASessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, QASession] = {}

    def create(self, file_id: str, questions: list[str], targets: list[dict]) -> QASession:
        session = QASession(
            session_id=uuid.uuid4().hex,
            file_id=file_id,
            questions=questions,
            targets=targets,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> QASession | None:
        return self._sessions.get(session_id)

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions
