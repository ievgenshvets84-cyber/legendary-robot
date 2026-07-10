"""Интеграционные тесты: полный стек против поддельного сервера Ollama.

В отличие от test_core.py (стабы), здесь поднимается настоящий HTTP-сервер,
эмулирующий API Ollama (/api/tags, /api/chat, /api/embeddings), и через него
прогоняется весь путь: LLMClient → Agent → инструменты → память. Так
проверяется транспорт, разбор ответов и обработка ошибок (включая 404
«модель не установлена») без реального Ollama.

Запуск:  python -m unittest tests.test_integration
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from localmind.agent import Agent
from localmind.llm import LLMClient, LLMError
from localmind.memory import Memory
from localmind.safety import Guard
from localmind.skill_manager import SkillManager
from localmind.tools import ToolContext, ToolRegistry


KNOWN_MODEL = "fake-model"


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Мини-эмулятор API Ollama. Ответы чата задаются очередью на сервере."""

    def log_message(self, *args) -> None:
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._json({"models": [{"name": KNOWN_MODEL},
                                   {"name": "fake-embed"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/api/chat":
            if payload.get("model") != KNOWN_MODEL:
                self._json({"error": f"model '{payload.get('model')}' not found, "
                                     "try pulling it first"}, 404)
                return
            queue = getattr(self.server, "chat_responses", [])
            content = queue.pop(0) if queue else json.dumps(
                {"thought": "готово", "final": "ok"})
            self.server.chat_requests.append(payload)
            self._json({"message": {"role": "assistant", "content": content}})
        elif self.path == "/api/embeddings":
            # Детерминированный «эмбеддинг» из длины и суммы кодов символов,
            # чтобы одинаковые тексты совпадали, а разные отличались.
            text = payload.get("prompt", "")
            vec = [len(text) % 17 + 1.0,
                   sum(ord(c) for c in text) % 101 + 1.0,
                   len(text.split()) + 1.0]
            self._json({"embedding": vec})
        else:
            self._json({"error": "not found"}, 404)


class FakeOllama:
    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        self.server.chat_responses = []
        self.server.chat_requests = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeOllama":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def queue(self, *contents: str) -> None:
        self.server.chat_responses.extend(contents)


class TestLLMClientIntegration(unittest.TestCase):
    def test_available_and_models(self):
        with FakeOllama() as ollama:
            llm = LLMClient(host=ollama.host, model=KNOWN_MODEL)
            self.assertTrue(llm.available())
            self.assertIn(KNOWN_MODEL, llm.installed_models())

    def test_chat_roundtrip(self):
        with FakeOllama() as ollama:
            ollama.queue("Привет из модели!")
            llm = LLMClient(host=ollama.host, model=KNOWN_MODEL)
            reply = llm.chat([{"role": "user", "content": "привет"}])
            self.assertEqual(reply, "Привет из модели!")
            # Параметры дошли до сервера.
            req = ollama.server.chat_requests[0]
            self.assertEqual(req["model"], KNOWN_MODEL)
            self.assertIn("options", req)

    def test_missing_model_gives_pull_hint(self):
        with FakeOllama() as ollama:
            llm = LLMClient(host=ollama.host, model="not-installed:7b")
            with self.assertRaises(LLMError) as ctx:
                llm.chat([{"role": "user", "content": "hi"}])
            msg = str(ctx.exception)
            self.assertIn("не установлена", msg)
            self.assertIn("ollama pull not-installed:7b", msg)
            self.assertIn("not found", msg)  # тело ответа Ollama проброшено

    def test_embeddings(self):
        with FakeOllama() as ollama:
            llm = LLMClient(host=ollama.host, model=KNOWN_MODEL,
                            embed_model="fake-embed")
            vec = llm.embed("тестовый текст")
            self.assertEqual(len(vec), 3)


class TestAgentIntegration(unittest.TestCase):
    """Полный цикл агента через HTTP: мысль → инструмент → наблюдение → финал."""

    def _build(self, workdir: Path, ollama: FakeOllama):
        llm = LLMClient(host=ollama.host, model=KNOWN_MODEL,
                        embed_model="fake-embed")
        guard = Guard(workdir)
        memory = Memory(workdir / "memory.jsonl", llm, min_score=0.01)
        ctx = ToolContext(guard=guard, memory=memory, llm=llm, sandbox_timeout=10)
        registry = ToolRegistry(ctx)
        skills = SkillManager(workdir / "skills", registry, guard, llm)
        agent = Agent(llm, registry, memory, skills, max_steps=6, verbose=False)
        return agent, memory

    def test_agent_writes_file_over_http(self):
        with tempfile.TemporaryDirectory() as d, FakeOllama() as ollama:
            agent, _ = self._build(Path(d), ollama)
            ollama.queue(
                json.dumps({"thought": "создаю файл", "tool": "write_file",
                            "args": {"path": "hello.txt", "content": "мир"}}),
                json.dumps({"thought": "проверяю", "tool": "read_file",
                            "args": {"path": "hello.txt"}}),
                json.dumps({"thought": "готово", "final": "файл создан и проверен"}),
            )
            result = agent.run("создай файл hello.txt")
            self.assertEqual(result, "файл создан и проверен")
            self.assertEqual((Path(d) / "hello.txt").read_text(encoding="utf-8"), "мир")

    def test_agent_recovers_from_invalid_json(self):
        with tempfile.TemporaryDirectory() as d, FakeOllama() as ollama:
            agent, _ = self._build(Path(d), ollama)
            ollama.queue(
                "просто текст без JSON — протокол нарушен",
                json.dumps({"thought": "исправился", "final": "ответ"}),
            )
            result = agent.run("что-нибудь")
            self.assertEqual(result, "ответ")

    def test_agent_unknown_tool_feedback(self):
        with tempfile.TemporaryDirectory() as d, FakeOllama() as ollama:
            agent, _ = self._build(Path(d), ollama)
            ollama.queue(
                json.dumps({"thought": "пробую", "tool": "no_such_tool", "args": {}}),
                json.dumps({"thought": "понял", "final": "использую другой путь"}),
            )
            result = agent.run("задача")
            self.assertEqual(result, "использую другой путь")
            # Модель получила наблюдение со списком доступных инструментов.
            last_user = [m for m in ollama.server.chat_requests[-1]["messages"]
                         if m["role"] == "user"][-1]
            self.assertIn("не существует", last_user["content"])

    def test_memory_semantic_roundtrip(self):
        with tempfile.TemporaryDirectory() as d, FakeOllama() as ollama:
            _, memory = self._build(Path(d), ollama)
            memory.add("столица Франции — Париж", kind="fact")
            memory.add("у кошки четыре лапы", kind="fact")
            hits = memory.search("столица Франции — Париж")
            self.assertTrue(hits)
            self.assertIn("Париж", hits[0]["text"])
            stats = memory.stats()
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["embedded"], 2)  # эмбеддинги реально пришли по HTTP


class TestWebIntegration(unittest.TestCase):
    """Веб-обработчики против настоящего HTTP-бэкенда."""

    def test_handle_chat_over_http(self):
        from localmind import web
        with tempfile.TemporaryDirectory() as d, FakeOllama() as ollama:
            ollama.queue("Ответ веб-чата")
            llm = LLMClient(host=ollama.host, model=KNOWN_MODEL,
                            embed_model="fake-embed")
            guard = Guard(Path(d))
            ctx = ToolContext(guard=guard, memory=None, llm=llm)
            registry = ToolRegistry(ctx)
            memory = Memory(Path(d) / "m.jsonl", llm)

            class StubCfg:
                llm = {"host": ollama.host, "model": KNOWN_MODEL}

            class App:
                pass

            app = App()
            app.llm = llm
            app.memory = memory
            app.registry = registry
            app.loaded_skills = []
            app.cfg = StubCfg()
            state = web.new_state()
            out = web.handle_chat(app, state, {"message": "привет"})
            self.assertEqual(out["reply"], "Ответ веб-чата")

            st = web.status_payload(app)
            self.assertTrue(st["ollama"])
            self.assertTrue(st["model_installed"])

    def test_history_trim(self):
        from localmind import web
        state = web.new_state()
        for i in range(web.MAX_HISTORY_TURNS + 20):
            state["history"].append({"role": "user", "content": f"m{i}"})
        web._trim_history(state)
        self.assertEqual(len(state["history"]), web.MAX_HISTORY_TURNS + 1)
        self.assertEqual(state["history"][0]["role"], "system")  # система на месте
        self.assertEqual(state["history"][-1]["content"],
                         f"m{web.MAX_HISTORY_TURNS + 19}")  # свежие сохранены


if __name__ == "__main__":
    unittest.main()
