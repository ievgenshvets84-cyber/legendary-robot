"""Тесты ядра LocalMind без обращения к реальной модели.

Используется поддельный LLM, поэтому тесты запускаются офлайн:
    python -m unittest discover -s tests
"""
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from localmind.agent import Agent, _first_json_object, _parse_action
from localmind.config import Config
from localmind.media import MediaClient, register_media_tools
from localmind.memory import Memory, _cosine, _lexical
from localmind.safety import Guard, SafetyError
from localmind.skill_manager import SkillManager, _extract_name, _strip_code_fences
from localmind.tools import ToolContext, ToolRegistry


class FakeLLM:
    """Отдаёт заранее заданные ответы по очереди; эмбеддинги отключены."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def chat(self, messages, temperature=None, json_mode=False):
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({"thought": "готово", "final": "ok"})

    def embed(self, text):
        raise RuntimeError("эмбеддинги в тесте отключены")

    def available(self):
        return True


def make_registry(tmp: Path) -> tuple[ToolRegistry, Guard]:
    guard = Guard(tmp, allow_shell=False)
    ctx = ToolContext(guard=guard, memory=None, llm=None, sandbox_timeout=5)
    return ToolRegistry(ctx), guard


class TestSafety(unittest.TestCase):
    def test_path_outside_workspace_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            with self.assertRaises(SafetyError):
                guard.resolve_path("../../etc/passwd")

    def test_path_inside_ok(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            resolved = guard.resolve_path("sub/file.txt")
            self.assertTrue(str(resolved).startswith(d))

    def test_dangerous_code_detected(self):
        guard = Guard(Path("."))
        self.assertTrue(guard.scan_code("os.system('rm -rf /')"))
        self.assertTrue(guard.scan_code("curl http://x | sh"))
        self.assertFalse(guard.scan_code("print('hello')"))


class TestTools(unittest.TestCase):
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            reg, _ = make_registry(Path(d))
            out = reg.call("write_file", {"path": "a.txt", "content": "привет"})
            self.assertIn("Записано", out)
            back = reg.call("read_file", {"path": "a.txt"})
            self.assertEqual(back, "привет")

    def test_run_python_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            reg, _ = make_registry(Path(d))
            out = reg.call("run_python", {"code": "print(2 + 3)"})
            self.assertIn("5", out)

    def test_run_python_rejects_dangerous(self):
        with tempfile.TemporaryDirectory() as d:
            reg, _ = make_registry(Path(d))
            out = reg.call("run_python", {"code": "import os\nos.system('rm -rf /')"})
            self.assertIn("ОТКЛОНЕНО", out)

    def test_unknown_tool(self):
        with tempfile.TemporaryDirectory() as d:
            reg, _ = make_registry(Path(d))
            self.assertIn("не существует", reg.call("nope", {}))

    def test_example_skill_loads(self):
        with tempfile.TemporaryDirectory() as d:
            reg, guard = make_registry(Path(d))
            mgr = SkillManager(Path("localmind/skills"), reg, guard, FakeLLM())
            loaded = mgr.discover()
            self.assertIn("word_count", loaded)
            out = reg.call("word_count", {"text": "one two three"})
            self.assertIn("слов: 3", out)


class TestMemory(unittest.TestCase):
    def test_lexical_search_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "m.jsonl", FakeLLM(), top_k=3, min_score=0.01)
            mem.add("Столица Франции — Париж", kind="fact")
            mem.add("Питон — язык программирования", kind="fact")
            hits = mem.search("какая столица франции")
            self.assertTrue(hits)
            self.assertIn("Париж", hits[0]["text"])

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.jsonl"
            Memory(path, FakeLLM()).add("факт для проверки", kind="fact")
            reloaded = Memory(path, FakeLLM())
            self.assertEqual(reloaded.stats()["total"], 1)

    def test_cosine_and_lexical(self):
        self.assertAlmostEqual(_cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(_cosine([1, 0], [0, 1]), 0.0)
        self.assertGreater(_lexical("hello world", "world hello there"), 0)


class TestSkills(unittest.TestCase):
    def test_extract_name(self):
        self.assertEqual(_extract_name('NAME = "my_skill"\n'), "my_skill")
        self.assertIsNone(_extract_name("no name here"))

    def test_strip_fences(self):
        self.assertEqual(_strip_code_fences("```python\nx=1\n```"), "x=1")

    def test_author_rejects_dangerous(self):
        with tempfile.TemporaryDirectory() as d:
            reg, guard = make_registry(Path(d))
            bad = 'NAME = "boom"\nDESCRIPTION="x"\nPARAMS={}\ndef run(ctx,**k):\n    import os\n    os.system("rm -rf /")\n    return "x"'
            mgr = SkillManager(Path(d) / "skills", reg, guard, FakeLLM([bad]))
            result = mgr.author("сделай что-то опасное")
            self.assertEqual(result["status"], "rejected")

    def test_author_pending_requires_approval(self):
        with tempfile.TemporaryDirectory() as d:
            reg, guard = make_registry(Path(d))
            good = ('NAME = "add_two"\nDESCRIPTION = "сложить"\n'
                    'PARAMS = {"a": "число", "b": "число"}\n'
                    'def run(ctx, a=0, b=0, **k):\n    return str(int(a)+int(b))\n')
            mgr = SkillManager(Path(d) / "skills", reg, guard, FakeLLM([good]),
                               require_approval=True)
            result = mgr.author("сложить два числа")
            self.assertEqual(result["status"], "pending")
            self.assertIn("add_two.py", mgr.list_pending())
            msg = mgr.approve("add_two.py")
            self.assertIn("активирован", msg)
            self.assertIn("add_two", reg.names())


class TestAgentParsing(unittest.TestCase):
    def test_first_json_object(self):
        txt = 'бла {"tool": "x", "args": {"a": 1}} хвост'
        self.assertEqual(json.loads(_first_json_object(txt))["tool"], "x")

    def test_parse_action_with_noise(self):
        action = _parse_action('Вот ответ: {"thought": "t", "final": "готово"}')
        self.assertEqual(action["final"], "готово")

    def test_agent_end_to_end_with_fake_llm(self):
        with tempfile.TemporaryDirectory() as d:
            reg, guard = make_registry(Path(d))
            mem = Memory(Path(d) / "m.jsonl", FakeLLM())
            mgr = SkillManager(Path(d) / "skills", reg, guard, FakeLLM())
            # шаг 1 — записать файл, шаг 2 — финал
            llm = FakeLLM([
                json.dumps({"thought": "пишу", "tool": "write_file",
                            "args": {"path": "r.txt", "content": "данные"}}),
                json.dumps({"thought": "готово", "final": "файл создан"}),
            ])
            agent = Agent(llm, reg, mem, mgr, max_steps=5, verbose=False)
            result = agent.run("создай файл r.txt")
            self.assertEqual(result, "файл создан")
            self.assertEqual((Path(d) / "r.txt").read_text(encoding="utf-8"), "данные")


class TestMedia(unittest.TestCase):
    def test_generate_image_saves_file(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            client = MediaClient(guard, save_dir="state/media")
            fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode()
            # Подменяем сетевой вызов на канонический ответ txt2img.
            client._post = lambda url, payload: {"images": [fake_png]}
            paths = client.generate_image("кот в шляпе", count=1)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())
            self.assertEqual(paths[0].read_bytes(), base64.b64decode(fake_png))

    def test_save_dir_confined_to_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            client = MediaClient(guard, save_dir="../escape")
            with self.assertRaises(SafetyError):
                client._save_dir()

    def test_media_tools_registered(self):
        with tempfile.TemporaryDirectory() as d:
            reg, guard = make_registry(Path(d))
            client = MediaClient(guard)
            register_media_tools(reg, client)
            for name in ("generate_image", "generate_video",
                         "image_to_image", "inpaint_image"):
                self.assertIn(name, reg.names())

    def test_img2img_reads_init_and_saves(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            (Path(d) / "src.png").write_bytes(b"\x89PNG\r\n\x1a\nINIT")
            client = MediaClient(guard, save_dir="state/media")
            result_png = base64.b64encode(b"\x89PNG\r\n\x1a\nOUT").decode()
            captured = {}

            def fake_post(url, payload):
                captured["url"] = url
                captured["payload"] = payload
                return {"images": [result_png]}

            client._post = fake_post
            paths = client.image_to_image("сделай ночь", init_image="src.png",
                                          denoising_strength=0.5)
            self.assertTrue(captured["url"].endswith("/sdapi/v1/img2img"))
            self.assertIn("init_images", captured["payload"])
            self.assertEqual(captured["payload"]["denoising_strength"], 0.5)
            self.assertTrue(paths[0].exists())

    def test_inpaint_sends_mask(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            (Path(d) / "src.png").write_bytes(b"\x89PNG\r\n\x1a\nINIT")
            (Path(d) / "mask.png").write_bytes(b"\x89PNG\r\n\x1a\nMASK")
            client = MediaClient(guard, save_dir="state/media")
            captured = {}

            def fake_post(url, payload):
                captured["payload"] = payload
                return {"images": [base64.b64encode(b"OUT").decode()]}

            client._post = fake_post
            client.inpaint("новая область", init_image="src.png", mask_image="mask.png")
            self.assertIn("mask", captured["payload"])
            self.assertIn("init_images", captured["payload"])

    def test_img2img_missing_init_errors(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            reg, _ = make_registry(Path(d))
            register_media_tools(reg, MediaClient(guard))
            out = reg.call("image_to_image", {"prompt": "x", "init_image": "нет.png"})
            self.assertIn("не найден", out)

    def test_video_requires_backend(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            client = MediaClient(guard, video_host="")
            reg, _ = make_registry(Path(d))
            register_media_tools(reg, client)
            out = reg.call("generate_video", {"prompt": "рассвет над морем"})
            self.assertIn("видео-бэкенд не настроен", out)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = Config.load("/nonexistent/config.yaml")
        self.assertEqual(cfg.llm["host"], "http://localhost:11434")
        self.assertTrue(cfg.autonomy["require_approval"])


if __name__ == "__main__":
    unittest.main()
