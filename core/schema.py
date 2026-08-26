"""Общий контракт вердикта для всех трёх кейсов.

Замороженный интерфейс: менять только по согласованию, на него завязаны все плагины.
Кейс-специфика уходит в `artifacts`, а не в новые поля верхнего уровня.
"""

from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["pass", "block", "manual_review"]

# Человекочитаемые названия решений для выгрузки в xlsx (кейсы формулируют по-русски).
ACTION_RU: dict[str, str] = {
    "pass": "пропустить",
    "block": "заблокировать",
    "manual_review": "отправить на ручную проверку",
}


class Verdict(BaseModel):
    doc_id: str
    verdict: str
    """Кейс-специфичное значение: safe|injection_malicious | vulnerable|secure|uncertain."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Откалиброванная уверенность, а не самооценка модели."""

    action: Action
    evidence: list[str] = Field(default_factory=list)
    """Теги из фиксированной таксономии кейса, не свободный текст."""

    rationale: str = ""
    """Одно-два предложения для человека."""

    artifacts: dict = Field(default_factory=dict)
    """Кейс-специфика: анонимизированный текст, найденные PII, патч, CWE и т.п."""
