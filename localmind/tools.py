"""Инструменты агента и их реестр.

Инструмент — это способность агента воздействовать на мир: читать и писать
файлы, исполнять код в песочнице, обращаться к памяти. Навыки (skills.py)
регистрируются здесь же как динамические инструменты, поэтому агент может
расширять собственный набор возможностей во время работы.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .safety import Guard, SafetyError


@dataclass
class ToolContext:
    """Общий контекст, доступный каждому инструменту при вызове."""
    guard: Guard
    memory: Any = None       # localmind.memory.Memory
    llm: Any = None          # localmind.llm.LLMClient
    sandbox_timeout: int = 20


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]                       # имя параметра -> пояснение
    func: Callable[..., str]
    dynamic: bool = False                        # добавлен ли как навык на лету

    def spec(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.params.items()) or "нет"
        tag = " [навык]" if self.dynamic else ""
        return f"- {self.name}({args}) — {self.description}{tag}"


class ToolRegistry:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self._tools: dict[str, Tool] = {}
        self._register_builtins()

    # ── доступ ───────────────────────────────────────────────────────────
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> str:
        return "\n".join(t.spec() for t in self._tools.values())

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"ОШИБКА: инструмента '{name}' не существует. Доступны: {', '.join(self.names())}"
        try:
            return tool.func(self.ctx, **args)
        except SafetyError as exc:
            return f"ОТКЛОНЕНО ПРЕДОХРАНИТЕЛЕМ: {exc}"
        except TypeError as exc:
            return f"ОШИБКА аргументов для '{name}': {exc}. Ожидались: {tool.params}"
        except Exception as exc:  # инструмент не должен ронять цикл агента
            return f"ОШИБКА при вызове '{name}': {type(exc).__name__}: {exc}"

    # ── встроенные инструменты ───────────────────────────────────────────
    def _register_builtins(self) -> None:
        self.register(Tool(
            "read_file", "Прочитать текстовый файл из рабочей папки.",
            {"path": "относительный путь к файлу"}, _read_file))
        self.register(Tool(
            "write_file", "Создать или перезаписать файл в рабочей папке.",
            {"path": "путь", "content": "содержимое"}, _write_file))
        self.register(Tool(
            "list_dir", "Показать содержимое каталога.",
            {"path": "путь (по умолчанию '.')"}, _list_dir))
        self.register(Tool(
            "run_python", "Исполнить фрагмент Python в изолированном процессе "
            "с тайм-аутом и вернуть stdout/stderr.",
            {"code": "исходный код Python"}, _run_python))
        self.register(Tool(
            "remember", "Записать факт или вывод в долговременную память.",
            {"text": "что запомнить", "tags": "список тегов через запятую (необязательно)"},
            _remember))
        self.register(Tool(
            "recall", "Найти релевантные воспоминания по запросу.",
            {"query": "поисковый запрос"}, _recall))
        self.register(Tool(
            "shell", "Выполнить shell-команду (по умолчанию отключено в config).",
            {"command": "команда"}, _shell))


# ── реализации встроенных инструментов ──────────────────────────────────
def _read_file(ctx: ToolContext, path: str) -> str:
    target = ctx.guard.resolve_path(path)
    if not target.exists():
        return f"Файл не найден: {path}"
    return target.read_text(encoding="utf-8", errors="replace")[:20000]


def _write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = ctx.guard.resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Записано {len(content)} символов в {path}"


def _list_dir(ctx: ToolContext, path: str = ".") -> str:
    target = ctx.guard.resolve_path(path)
    if not target.exists():
        return f"Каталог не найден: {path}"
    items = []
    for p in sorted(target.iterdir()):
        marker = "/" if p.is_dir() else ""
        items.append(f"{p.name}{marker}")
    return "\n".join(items) or "(пусто)"


def _run_python(ctx: ToolContext, code: str) -> str:
    ctx.guard.assert_code_safe(code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=ctx.guard.workspace, encoding="utf-8") as fh:
        fh.write(code)
        script = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True,
            timeout=ctx.sandbox_timeout, cwd=ctx.guard.workspace,
        )
        out = (proc.stdout or "")[:8000]
        err = (proc.stderr or "")[:4000]
        result = f"exit={proc.returncode}\n"
        if out:
            result += f"STDOUT:\n{out}\n"
        if err:
            result += f"STDERR:\n{err}"
        return result.strip()
    except subprocess.TimeoutExpired:
        return f"Исполнение прервано по тайм-ауту ({ctx.sandbox_timeout} c)."
    finally:
        Path(script).unlink(missing_ok=True)


def _remember(ctx: ToolContext, text: str, tags: str = "") -> str:
    if ctx.memory is None:
        return "Память недоступна."
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    ctx.memory.add(text, tags=tag_list, kind="fact")
    return "Запомнено."


def _recall(ctx: ToolContext, query: str) -> str:
    if ctx.memory is None:
        return "Память недоступна."
    hits = ctx.memory.search(query)
    if not hits:
        return "Ничего релевантного не найдено."
    return "\n".join(f"- {h['text']}" for h in hits)


def _shell(ctx: ToolContext, command: str) -> str:
    ctx.guard.assert_shell_allowed()
    ctx.guard.assert_code_safe(command)
    proc = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        timeout=ctx.sandbox_timeout, cwd=ctx.guard.workspace,
    )
    return f"exit={proc.returncode}\nSTDOUT:\n{proc.stdout[:6000]}\nSTDERR:\n{proc.stderr[:2000]}"
