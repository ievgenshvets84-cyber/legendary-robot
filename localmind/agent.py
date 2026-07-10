"""Ядро агента: цикл рассуждение → действие → наблюдение (ReAct).

Работает с любой инструктивной открытой моделью, потому что протокол
взаимодействия — это JSON, который модель выдаёт на каждом шаге. Так агент
пользуется инструментами, обращается к памяти и, при необходимости, создаёт
себе новые навыки.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .llm import LLMClient
from .memory import Memory
from .skill_manager import SkillManager
from .tools import Tool, ToolRegistry


SYSTEM_PROMPT = """Ты — LocalMind, автономный ИИ-агент, работающий локально.
Ты решаешь задачу пошагово, пользуясь инструментами.

На КАЖДОМ шаге отвечай СТРОГО одним JSON-объектом одного из двух видов:

1) Вызов инструмента:
{{"thought": "краткое рассуждение", "tool": "имя_инструмента", "args": {{...}}}}

2) Итоговый ответ, когда задача решена:
{{"thought": "почему готово", "final": "ответ пользователю"}}

Никакого текста вне JSON. Не выдумывай инструменты, которых нет в списке.
Если не хватает способности — создай её инструментом create_skill, затем используй.

Доступные инструменты:
{catalog}

{memory}"""


class Agent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, memory: Memory,
                 skill_manager: SkillManager, max_steps: int = 12,
                 verbose: bool = True) -> None:
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.skills = skill_manager
        self.max_steps = max_steps
        self.verbose = verbose
        self._register_meta_tools()

    def _register_meta_tools(self) -> None:
        def _create_skill(ctx, description: str) -> str:
            result = self.skills.author(description)
            status = result.get("status")
            if status == "created":
                return f"Навык '{result['name']}' создан и доступен."
            if status == "pending":
                return (f"Навык '{result['name']}' написан, но ждёт одобрения человека "
                        f"(файл {result['file']}). До одобрения он недоступен.")
            return f"Навык не создан: {result.get('reason', 'неизвестно')}"

        self.registry.register(Tool(
            "create_skill",
            "Написать себе новый навык (инструмент) по текстовому описанию задачи.",
            {"description": "что должен делать навык"},
            _create_skill,
        ))

    # ── основной цикл ────────────────────────────────────────────────────
    def run(self, task: str, on_event: Callable[[str, str], None] | None = None) -> str:
        recall = self.memory.recall_block(task)
        system = SYSTEM_PROMPT.format(catalog=self.registry.catalog(), memory=recall)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        def emit(kind: str, text: str) -> None:
            if on_event:
                on_event(kind, text)
            elif self.verbose:
                print(f"[{kind}] {text}")

        last_observation = ""
        for step in range(1, self.max_steps + 1):
            raw = self.llm.chat(messages, json_mode=True)
            action = _parse_action(raw)
            messages.append({"role": "assistant", "content": raw})

            if action is None:
                emit("warn", "не удалось разобрать ответ модели как JSON")
                messages.append({"role": "user", "content":
                                 "Ответ невалиден. Верни строго JSON по протоколу."})
                continue

            thought = action.get("thought", "")
            if thought:
                emit("thought", thought)

            if "final" in action:
                final = str(action["final"])
                self.memory.add(f"Задача: {task}\nРешение: {final}", kind="episode",
                                tags=["episode"])
                emit("final", final)
                return final

            tool_name = action.get("tool", "")
            args = action.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            emit("action", f"{tool_name}({json.dumps(args, ensure_ascii=False)})")
            observation = self.registry.call(tool_name, args)
            last_observation = observation
            emit("observation", _truncate(observation, 600))
            messages.append({"role": "user",
                             "content": f"Наблюдение:\n{observation}"})

        emit("warn", f"достигнут предел шагов ({self.max_steps})")
        return last_observation or "Не удалось завершить задачу за отведённые шаги."


# ── разбор ответа модели ────────────────────────────────────────────────
def _parse_action(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    # Прямой разбор.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Извлечение первого сбалансированного JSON-объекта из текста.
    match = _first_json_object(raw)
    if match:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " …"
