"""Автономный режим: работа по списку целей без участия человека.

Агент читает цели из markdown-файла (state/goals.md), выполняет их по одной,
отмечает выполненные и записывает итог. После проработки списка он может
отрефлексировать сделанное и предложить новые цели — но НЕ выполняет их
автоматически без явного разрешения, чтобы автономность оставалась
управляемой.
"""
from __future__ import annotations

import time
from pathlib import Path

from .agent import Agent
from .llm import LLMClient
from .memory import Memory


class AutonomyLoop:
    def __init__(self, agent: Agent, memory: Memory, llm: LLMClient,
                 goal_file: str | Path, max_cycles: int = 25,
                 propose_new: bool = False) -> None:
        self.agent = agent
        self.memory = memory
        self.llm = llm
        self.goal_file = Path(goal_file)
        self.goal_file.parent.mkdir(parents=True, exist_ok=True)
        self.max_cycles = max_cycles
        self.propose_new = propose_new

    # ── работа с целями ──────────────────────────────────────────────────
    def _read_goals(self) -> list[tuple[bool, str]]:
        if not self.goal_file.exists():
            return []
        goals: list[tuple[bool, str]] = []
        for line in self.goal_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("- [ ]"):
                goals.append((False, s[5:].strip()))
            elif s.startswith("- [x]") or s.startswith("- [X]"):
                goals.append((True, s[5:].strip()))
        return goals

    def _write_goals(self, goals: list[tuple[bool, str]]) -> None:
        lines = ["# Цели LocalMind", ""]
        for done, text in goals:
            box = "x" if done else " "
            lines.append(f"- [{box}] {text}")
        self.goal_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def add_goal(self, text: str) -> None:
        goals = self._read_goals()
        goals.append((False, text))
        self._write_goals(goals)

    # ── цикл ─────────────────────────────────────────────────────────────
    def run(self) -> str:
        completed = 0
        for cycle in range(1, self.max_cycles + 1):
            goals = self._read_goals()
            pending = [g for g in goals if not g[0]]
            if not pending:
                print("[autonomy] Все цели выполнены.")
                if self.propose_new:
                    self._propose_next_goals(goals)
                break

            _, goal_text = pending[0]
            print(f"\n[autonomy] Цикл {cycle}: цель → {goal_text}")
            result = self.agent.run(goal_text)

            # Отметить цель выполненной.
            new_goals: list[tuple[bool, str]] = []
            marked = False
            for done, text in goals:
                if not marked and not done and text == goal_text:
                    new_goals.append((True, text))
                    marked = True
                else:
                    new_goals.append((done, text))
            self._write_goals(new_goals)
            self.memory.add(f"Автономно выполнена цель: {goal_text}\nИтог: {result}",
                            kind="episode", tags=["autonomy"])
            completed += 1
            print(f"[autonomy] Готово: {goal_text}")
            time.sleep(0.2)

        return f"Автономный прогон завершён. Выполнено целей: {completed}."

    def _propose_next_goals(self, goals: list[tuple[bool, str]]) -> None:
        """Рефлексия: модель предлагает новые цели (только как черновик)."""
        done_list = "\n".join(f"- {t}" for _, t in goals)
        prompt = (
            "Ты только что выполнил список задач:\n" + done_list +
            "\n\nПредложи 3 осмысленные следующие цели для развития проекта. "
            "Ответ — просто список строк, по одной цели на строку."
        )
        try:
            suggestion = self.llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            print(f"[autonomy] Не удалось предложить новые цели: {exc}")
            return
        draft = self.goal_file.with_name("proposed_goals.md")
        draft.write_text("# Предложенные цели (черновик, требует одобрения)\n\n"
                         + suggestion + "\n", encoding="utf-8")
        print(f"[autonomy] Новые цели предложены в {draft.name} — одобрите вручную.")
