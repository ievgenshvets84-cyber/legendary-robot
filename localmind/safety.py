"""Защитные механизмы LocalMind.

Автономный самопрограммирующийся агент без ограничений опасен: он может
затереть данные, выйти за пределы рабочей папки или сгенерировать
разрушительный код. Этот модуль ограничивает область действия и отсеивает
явно опасные конструкции ещё до исполнения.
"""
from __future__ import annotations

import re
from pathlib import Path


class SafetyError(RuntimeError):
    """Операция отклонена предохранителем."""


# Паттерны, которые почти всегда означают разрушительное действие.
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-rf\s+/(?:\s|$)", "удаление корня файловой системы"),
    (r"rm\s+-rf\s+~", "удаление домашней папки"),
    (r":\(\)\s*\{.*\};\s*:", "fork-бомба"),
    (r"\bmkfs\b", "форматирование раздела"),
    (r"\bdd\s+if=.*of=/dev/", "запись напрямую в устройство"),
    (r">\s*/dev/sd[a-z]", "запись в блочное устройство"),
    (r"shutil\.rmtree\(\s*['\"]/['\"]", "рекурсивное удаление корня"),
    (r"os\.system\(\s*['\"]?\s*rm\s+-rf", "os.system с rm -rf"),
    (r"\bchmod\s+-R\s+777\s+/", "снятие прав со всей системы"),
    (r"(curl|wget)\s+[^|]*\|\s*(sudo\s+)?(ba)?sh", "запуск скрипта из сети напрямую в shell"),
    (r"\bsudo\b", "повышение привилегий"),
    (r"crontab\s+-r", "очистка расписания задач"),
]


class Guard:
    def __init__(self, workspace: Path, allow_shell: bool = False) -> None:
        self.workspace = workspace.resolve()
        self.allow_shell = allow_shell

    # ── файловая песочница ───────────────────────────────────────────────
    def resolve_path(self, path: str) -> Path:
        """Приводит путь к абсолютному и требует, чтобы он был внутри workspace."""
        candidate = (self.workspace / path).resolve() if not Path(path).is_absolute() \
            else Path(path).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise SafetyError(
                f"Путь '{path}' выходит за пределы рабочей папки {self.workspace}. "
                "Операция запрещена."
            ) from exc
        return candidate

    # ── проверка кода/команд ─────────────────────────────────────────────
    def scan_code(self, code: str) -> list[str]:
        """Возвращает список сработавших опасных паттернов (пустой = чисто)."""
        hits: list[str] = []
        for pattern, description in DANGEROUS_PATTERNS:
            if re.search(pattern, code, flags=re.IGNORECASE):
                hits.append(description)
        return hits

    def assert_code_safe(self, code: str) -> None:
        hits = self.scan_code(code)
        if hits:
            raise SafetyError(
                "Код отклонён предохранителем. Обнаружено: " + "; ".join(hits)
            )

    def assert_shell_allowed(self) -> None:
        if not self.allow_shell:
            raise SafetyError(
                "Произвольный shell отключён (safety.allow_shell=false). "
                "Включите осознанно в config.yaml, если действительно нужно."
            )
