"""Самопрограммирование: агент пишет себе новые навыки.

Навык — это отдельный Python-модуль в папке skills/, который объявляет
NAME, DESCRIPTION, PARAMS и функцию run(ctx, **kwargs). Менеджер обнаруживает
такие модули, регистрирует их как инструменты, а также умеет ПОРОЖДАТЬ новые:
языковая модель пишет код навыка по описанию задачи, код проходит проверку
предохранителем и смоук-тест, и только потом становится доступной способностью.

Ключевой предохранитель: при autonomy.require_approval=true новый навык
кладётся в skills/pending/ и не активируется, пока человек его не одобрит
(перенос в skills/). Это и есть управляемое самоулучшение вместо
бесконтрольного переписывания себя.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .safety import Guard, SafetyError
from .tools import Tool, ToolContext, ToolRegistry


SKILL_TEMPLATE_HINT = '''\
Каждый навык — это ровно один Python-файл со следующим контрактом:

    NAME = "имя_навыка"              # латиница, snake_case, глагол
    DESCRIPTION = "что делает навык, одна строка"
    PARAMS = {"arg1": "пояснение", "arg2": "пояснение"}

    def run(ctx, **kwargs):
        # ctx.guard  — песочница путей (ctx.guard.resolve_path)
        # ctx.memory — долговременная память (может быть None)
        # ctx.llm    — доступ к языковой модели
        # Возвращать нужно СТРОКУ с результатом.
        ...
        return "результат"

Требования:
  * только стандартная библиотека Python, без внешних пакетов;
  * никаких сетевых запросов и разрушительных операций;
  * файловые операции только через ctx.guard.resolve_path(path);
  * функция run обязана вернуть строку.
'''


class SkillManager:
    def __init__(self, skills_dir: str | Path, registry: ToolRegistry,
                 guard: Guard, llm: LLMClient, require_approval: bool = True) -> None:
        self.dir = Path(skills_dir)
        self.pending = self.dir / "pending"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.guard = guard
        self.llm = llm
        self.require_approval = require_approval

    # ── загрузка существующих навыков ────────────────────────────────────
    def discover(self) -> list[str]:
        """Импортирует все навыки из skills/ и регистрирует как инструменты."""
        loaded: list[str] = []
        for file in sorted(self.dir.glob("*.py")):
            if file.name.startswith("_"):
                continue
            tool = self._load_module_as_tool(file)
            if tool:
                self.registry.register(tool)
                loaded.append(tool.name)
        return loaded

    def list_pending(self) -> list[str]:
        return [p.name for p in sorted(self.pending.glob("*.py"))]

    def approve(self, filename: str) -> str:
        """Переносит навык из pending/ в активную папку и регистрирует его."""
        src = self.pending / filename
        if not src.exists():
            return f"Нет ожидающего навыка: {filename}"
        dst = self.dir / filename
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        src.unlink()
        tool = self._load_module_as_tool(dst)
        if tool:
            self.registry.register(tool)
            return f"Навык '{tool.name}' одобрен и активирован."
        return f"Файл перенесён, но не является валидным навыком: {filename}"

    # ── порождение нового навыка ─────────────────────────────────────────
    def author(self, description: str) -> dict[str, Any]:
        """Просит модель написать навык по описанию, проверяет и сохраняет.

        Возвращает словарь со статусом: created | pending | rejected.
        """
        code = self._generate_code(description)
        code = _strip_code_fences(code)

        # 1. Проверка предохранителем на разрушительные конструкции.
        hits = self.guard.scan_code(code)
        if hits:
            return {"status": "rejected", "reason": "; ".join(hits), "code": code}

        # 2. Извлечение имени и базовая валидация контракта.
        name = _extract_name(code)
        if not name:
            return {"status": "rejected",
                    "reason": "в коде нет корректного NAME/контракта навыка",
                    "code": code}

        # 3. Смоук-тест: модуль должен импортироваться в отдельном процессе.
        ok, log = self._smoke_test(code)
        if not ok:
            return {"status": "rejected", "reason": f"смоук-тест не пройден: {log}",
                    "code": code}

        filename = f"{name}.py"
        header = f"# Навык сгенерирован LocalMind {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Задача: {description.strip()[:200]}\n\n"
        payload = header + code + "\n"

        # 4. Режим подтверждения: pending либо сразу активация.
        if self.require_approval:
            (self.pending / filename).write_text(payload, encoding="utf-8")
            return {"status": "pending", "name": name, "file": filename,
                    "hint": "Одобрите: localmind skills approve " + filename}

        target = self.dir / filename
        target.write_text(payload, encoding="utf-8")
        tool = self._load_module_as_tool(target)
        if tool:
            self.registry.register(tool)
            return {"status": "created", "name": name, "file": filename}
        return {"status": "rejected", "reason": "модуль не регистрируется как инструмент",
                "code": code}

    # ── внутреннее ───────────────────────────────────────────────────────
    def _generate_code(self, description: str) -> str:
        system = (
            "Ты пишешь навык-плагин для локального ИИ-агента. "
            "Ответ — ТОЛЬКО исходный код Python, без пояснений и без markdown.\n\n"
            + SKILL_TEMPLATE_HINT
        )
        user = f"Напиши навык, который решает задачу:\n{description}"
        return self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.2,
        )

    def _smoke_test(self, code: str) -> tuple[bool, str]:
        """Импортирует навык в изолированном процессе и проверяет контракт."""
        harness = textwrap.dedent(
            """
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("_skill_probe", sys.argv[1])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert isinstance(getattr(mod, "NAME", None), str) and mod.NAME
            assert isinstance(getattr(mod, "DESCRIPTION", None), str)
            assert isinstance(getattr(mod, "PARAMS", None), dict)
            assert callable(getattr(mod, "run", None))
            print("SKILL_OK")
            """
        )
        skill_file = self.pending / "_probe_candidate.py"
        harness_file = self.pending / "_probe_harness.py"
        skill_file.write_text(code, encoding="utf-8")
        harness_file.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(harness_file), str(skill_file)],
                capture_output=True, text=True, timeout=15,
            )
            if "SKILL_OK" in proc.stdout:
                return True, "ok"
            return False, (proc.stderr or proc.stdout)[:500]
        except subprocess.TimeoutExpired:
            return False, "тайм-аут импорта"
        finally:
            skill_file.unlink(missing_ok=True)
            harness_file.unlink(missing_ok=True)

    def _load_module_as_tool(self, file: Path) -> Tool | None:
        try:
            spec = importlib.util.spec_from_file_location(f"localmind_skill_{file.stem}", file)
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            name = getattr(module, "NAME", None)
            run = getattr(module, "run", None)
            if not isinstance(name, str) or not callable(run):
                return None
            description = getattr(module, "DESCRIPTION", "навык без описания")
            params = getattr(module, "PARAMS", {})
            return Tool(name=name, description=description, params=params,
                        func=run, dynamic=True)
        except Exception:
            return None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_name(code: str) -> str | None:
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("NAME"):
            _, _, value = stripped.partition("=")
            value = value.strip().strip('"').strip("'")
            if value and all(c.isalnum() or c == "_" for c in value):
                return value
    return None
