"""Клиент локальной языковой модели через Ollama.

Использует только стандартную библиотеку (urllib), чтобы ядро запускалось
без установки зависимостей. Модель работает полностью локально — данные
никуда не уходят. Совместимо с любой открытой моделью, доступной в Ollama
(llama3, qwen2.5, mistral, gemma, phi и т.д.).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Iterator


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        embed_model: str = "nomic-embed-text",
        temperature: float = 0.3,
        num_ctx: int = 8192,
        timeout: int = 300,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    # ── публичный API ────────────────────────────────────────────────────
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Синхронный ответ модели на список сообщений роли/контента."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Потоковый ответ по токенам — для интерактивного чата."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        for chunk in self._post_stream("/api/chat", payload):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece

    def embed(self, text: str) -> list[float]:
        """Векторное представление текста для семантической памяти."""
        payload = {"model": self.embed_model, "prompt": text}
        data = self._post("/api/embeddings", payload)
        vec = data.get("embedding")
        if not vec:
            raise LLMError("Пустой embedding — проверьте, что модель установлена: "
                           f"ollama pull {self.embed_model}")
        return list(vec)

    def available(self) -> bool:
        """Проверка, что локальный сервер Ollama поднят."""
        try:
            self._get("/api/tags")
            return True
        except Exception:
            return False

    def installed_models(self) -> list[str]:
        try:
            data = self._get("/api/tags")
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── транспорт ────────────────────────────────────────────────────────
    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.host + path, method="GET")
        return self._send(req)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._send(req)

    def _send(self, req: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Не удаётся связаться с Ollama на {self.host}. "
                "Запустите `ollama serve` и убедитесь, что модель загружена. "
                f"Причина: {exc}"
            ) from exc

    def _http_error(self, exc: "urllib.error.HTTPError") -> LLMError:
        """Различает «сервер не запущен» и «модель не установлена» (404)."""
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            pass
        detail = _extract_error(body)
        if exc.code == 404:
            return LLMError(
                f"Ollama запущен, но вернул 404 для модели '{self.model}'. "
                f"Чаще всего это значит, что модель не установлена — выполните:\n"
                f"    ollama pull {self.model}\n"
                + (f"Ответ сервера: {detail}" if detail else "")
            )
        return LLMError(
            f"Ollama вернул HTTP {exc.code}."
            + (f" Ответ: {detail}" if detail else "")
        )

    def _post_stream(self, path: str, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if line:
                        yield json.loads(line.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Ошибка потокового запроса к Ollama: {exc}") from exc


def _extract_error(body: str) -> str:
    """Достаёт поле error из JSON-ответа Ollama, иначе обрезанное тело."""
    body = (body or "").strip()
    if not body:
        return ""
    try:
        obj = json.loads(body)
        if isinstance(obj, dict) and obj.get("error"):
            return str(obj["error"])
    except json.JSONDecodeError:
        pass
    return body[:300]
