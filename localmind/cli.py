"""Командный интерфейс LocalMind.

Точка входа, которая связывает модель, память, инструменты, навыки и
автономный цикл. Запуск:

    python -m localmind doctor          # проверить окружение
    python -m localmind chat            # интерактивный диалог
    python -m localmind run "задача"    # выполнить одну задачу агентом
    python -m localmind auto            # автономный режим по state/goals.md
    python -m localmind skills list     # доступные навыки
    python -m localmind memory stats    # статистика памяти
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent
from .autonomy import AutonomyLoop
from .config import Config
from .llm import LLMClient
from .media import MediaClient, register_media_tools
from .memory import Memory
from .safety import Guard
from .skill_manager import SkillManager
from .tools import ToolContext, ToolRegistry


class App:
    """Собранный экземпляр агента со всеми подсистемами."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.llm = LLMClient(**cfg.llm)
        self.guard = Guard(cfg.workspace(), allow_shell=bool(cfg.safety["allow_shell"]))
        self.memory = Memory(cfg.root / cfg.memory["path"], self.llm,
                             top_k=int(cfg.memory["top_k"]),
                             min_score=float(cfg.memory["min_score"]))
        ctx = ToolContext(guard=self.guard, memory=self.memory, llm=self.llm,
                          sandbox_timeout=int(cfg.safety["sandbox_timeout"]))
        self.registry = ToolRegistry(ctx)
        self.media = MediaClient(
            self.guard, save_dir=cfg.media["save_dir"],
            image_host=cfg.media["image_host"], video_host=cfg.media["video_host"],
            steps=int(cfg.media["steps"]), width=int(cfg.media["width"]),
            height=int(cfg.media["height"]), cfg_scale=float(cfg.media["cfg_scale"]),
            sampler=str(cfg.media["sampler"]), timeout=int(cfg.media["timeout"]))
        register_media_tools(self.registry, self.media)
        self.skills = SkillManager(cfg.root / cfg.skills["dir"], self.registry,
                                   self.guard, self.llm,
                                   require_approval=bool(cfg.autonomy["require_approval"]))
        self.loaded_skills = self.skills.discover()
        self.agent = Agent(self.llm, self.registry, self.memory, self.skills,
                           max_steps=int(cfg.agent["max_steps"]),
                           verbose=bool(cfg.agent["verbose"]))


def build_app(config_path: str | None = None) -> App:
    return App(Config.load(config_path))


# ── подкоманды ───────────────────────────────────────────────────────────
def cmd_doctor(app: App, args: argparse.Namespace) -> int:
    print("LocalMind — проверка окружения\n" + "-" * 34)
    host = app.cfg.llm["host"]
    if app.llm.available():
        print(f"[ok] Ollama доступен: {host}")
        installed = app.llm.installed_models()
        print(f"     Установленные модели: {', '.join(installed) or '(нет)'}")
        target = app.cfg.llm["model"]
        if target in installed:
            print(f"[ok] Целевая модель установлена: {target}")
        else:
            print(f"[!]  Модель '{target}' не найдена. Установите: ollama pull {target}")
        if app.cfg.llm["embed_model"] not in installed:
            print(f"[!]  Модель эмбеддингов '{app.cfg.llm['embed_model']}' не найдена. "
                  f"Память будет работать в лексическом режиме.")
    else:
        print(f"[!]  Ollama недоступен на {host}. Запустите `ollama serve`.")
    print(f"\nРабочая папка (песочница): {app.guard.workspace}")
    print(f"Загружено навыков: {len(app.loaded_skills)} "
          f"({', '.join(app.loaded_skills) or 'нет'})")
    print(f"Инструментов всего: {len(app.registry.names())}")
    pending = app.skills.list_pending()
    if pending:
        print(f"Ждут одобрения навыки: {', '.join(pending)}")
    img = "доступен" if app.media.image_available() else "не запущен"
    vid = ("доступен" if app.media.video_available()
           else ("не настроен" if not app.cfg.media["video_host"] else "не запущен"))
    print(f"Медиа: изображения ({app.cfg.media['image_host']}) — {img}; видео — {vid}")
    print(f"Память: {app.memory.stats()}")
    return 0


def cmd_media(app: App, args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    neg = args.negative or ""
    if args.kind in ("image", "video", "img2img", "inpaint") and not prompt:
        print(f"Для режима '{args.kind}' нужен текстовый запрос.")
        return 1
    try:
        if args.kind == "image":
            paths = app.media.generate_image(prompt, negative_prompt=neg, count=args.count)
        elif args.kind == "video":
            paths = app.media.generate_video(prompt, negative_prompt=neg, seconds=args.seconds)
        elif args.kind == "img2img":
            if not args.init:
                print("Для img2img укажите --init <путь к изображению>.")
                return 1
            paths = app.media.image_to_image(prompt, init_image=args.init,
                                             negative_prompt=neg,
                                             denoising_strength=args.strength)
        elif args.kind == "inpaint":
            if not args.init or not args.mask:
                print("Для inpaint укажите --init <изображение> и --mask <маска>.")
                return 1
            paths = app.media.inpaint(prompt, init_image=args.init, mask_image=args.mask,
                                      negative_prompt=neg, denoising_strength=args.strength)
        elif args.kind == "img2video":
            if not args.init:
                print("Для img2video укажите --init <путь к изображению>.")
                return 1
            paths = app.media.image_to_video(init_image=args.init, prompt=prompt,
                                             seconds=args.seconds, fps=args.fps)
        elif args.kind == "slideshow":
            if not args.images:
                print("Для slideshow укажите --images путь1,путь2,...")
                return 1
            frames = [s.strip() for s in args.images.split(",") if s.strip()]
            paths = app.media.images_to_video(frames, fps=args.fps)
        else:
            print(f"Неизвестный режим: {args.kind}")
            return 1
    except Exception as exc:
        print(f"Ошибка генерации: {exc}")
        return 1
    print("Готово:")
    for p in paths:
        print(f"  {p}")
    return 0


def cmd_chat(app: App, args: argparse.Namespace) -> int:
    if not app.llm.available():
        print("Ollama недоступен. Сначала `ollama serve` и `ollama pull "
              f"{app.cfg.llm['model']}`.")
        return 1
    print("Диалог с LocalMind. Пустая строка или 'exit' — выход.\n")
    history: list[dict[str, str]] = [
        {"role": "system", "content":
         "Ты — LocalMind, локальный дружелюбный ассистент. Отвечай по-русски, кратко и по делу."}
    ]
    while True:
        try:
            user = input("вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in ("exit", "quit", "выход"):
            break
        recall = app.memory.recall_block(user)
        content = user if not recall else f"{recall}\n\nВопрос: {user}"
        history.append({"role": "user", "content": content})
        print("LocalMind> ", end="", flush=True)
        chunks: list[str] = []
        for piece in app.llm.stream(history):
            chunks.append(piece)
            print(piece, end="", flush=True)
        print("\n")
        answer = "".join(chunks)
        history.append({"role": "assistant", "content": answer})
        app.memory.add(f"Пользователь: {user}\nОтвет: {answer}", kind="dialog")
    return 0


def cmd_run(app: App, args: argparse.Namespace) -> int:
    if not app.llm.available():
        print("Ollama недоступен.")
        return 1
    task = " ".join(args.task)
    result = app.agent.run(task)
    print("\n=== РЕЗУЛЬТАТ ===")
    print(result)
    return 0


def cmd_auto(app: App, args: argparse.Namespace) -> int:
    if not app.llm.available():
        print("Ollama недоступен.")
        return 1
    goal_file = app.cfg.root / app.cfg.autonomy["goal_file"]
    if args.goal:
        loop = AutonomyLoop(app.agent, app.memory, app.llm, goal_file,
                            max_cycles=int(app.cfg.autonomy["max_cycles"]),
                            propose_new=args.propose)
        loop.add_goal(" ".join(args.goal))
    else:
        loop = AutonomyLoop(app.agent, app.memory, app.llm, goal_file,
                            max_cycles=int(app.cfg.autonomy["max_cycles"]),
                            propose_new=args.propose)
    if not goal_file.exists():
        print(f"Нет файла целей {goal_file}. Добавьте цели вида '- [ ] ...' "
              "или запустите с аргументом-целью.")
        return 1
    print(loop.run())
    return 0


def cmd_skills(app: App, args: argparse.Namespace) -> int:
    if args.action == "list":
        print("Активные навыки:")
        for name in app.loaded_skills or []:
            tool = app.registry.get(name)
            print(f"  {tool.spec() if tool else name}")
        pending = app.skills.list_pending()
        if pending:
            print("\nОжидают одобрения:")
            for p in pending:
                print(f"  {p}")
    elif args.action == "pending":
        for p in app.skills.list_pending():
            print(p)
    elif args.action == "approve":
        if not args.name:
            print("Укажите имя файла навыка.")
            return 1
        print(app.skills.approve(args.name))
    elif args.action == "new":
        desc = " ".join(args.name_parts) if args.name_parts else args.name or ""
        if not desc:
            print("Опишите, что должен делать навык.")
            return 1
        result = app.skills.author(desc)
        print(result)
    return 0


def cmd_memory(app: App, args: argparse.Namespace) -> int:
    if args.action == "stats":
        print(app.memory.stats())
    elif args.action == "search":
        for hit in app.memory.search(" ".join(args.query)):
            print(f"- ({hit['kind']}) {hit['text']}")
    elif args.action == "add":
        app.memory.add(" ".join(args.query), kind="fact")
        print("Добавлено в память.")
    return 0


# ── парсер аргументов ────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localmind",
                                description="Локальный автономный ИИ-агент на открытых моделях.")
    p.add_argument("--config", help="путь к config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="проверить окружение и модели")
    sub.add_parser("chat", help="интерактивный диалог")

    pr = sub.add_parser("run", help="выполнить одну задачу агентом")
    pr.add_argument("task", nargs="+")

    pa = sub.add_parser("auto", help="автономный режим по целям")
    pa.add_argument("goal", nargs="*", help="добавить и выполнить цель")
    pa.add_argument("--propose", action="store_true",
                    help="после выполнения предложить новые цели (черновик)")

    ps = sub.add_parser("skills", help="управление навыками")
    ps.add_argument("action", choices=["list", "pending", "approve", "new"])
    ps.add_argument("name", nargs="?", help="имя файла (approve)")
    ps.add_argument("name_parts", nargs="*", help="описание (new)")

    pm = sub.add_parser("memory", help="работа с памятью")
    pm.add_argument("action", choices=["stats", "search", "add"])
    pm.add_argument("query", nargs="*")

    pmd = sub.add_parser("media", help="локальная генерация изображений и видео")
    pmd.add_argument("kind", choices=["image", "video", "img2img", "inpaint",
                                      "img2video", "slideshow"])
    pmd.add_argument("prompt", nargs="*", help="описание сцены (для slideshow не нужно)")
    pmd.add_argument("--negative", help="что исключить из генерации")
    pmd.add_argument("--count", type=int, default=1, help="сколько изображений (1–4)")
    pmd.add_argument("--seconds", type=float, default=2.0, help="длительность видео")
    pmd.add_argument("--init", help="исходное изображение (img2img/inpaint/img2video)")
    pmd.add_argument("--mask", help="чёрно-белая маска (inpaint)")
    pmd.add_argument("--strength", type=float, default=0.6,
                     help="сила изменений 0.0–1.0 (img2img/inpaint)")
    pmd.add_argument("--images", help="кадры для slideshow: пути через запятую")
    pmd.add_argument("--fps", type=int, default=8, help="кадров в секунду (видео/slideshow)")
    return p


COMMANDS = {
    "doctor": cmd_doctor, "chat": cmd_chat, "run": cmd_run,
    "auto": cmd_auto, "skills": cmd_skills, "memory": cmd_memory,
    "media": cmd_media,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = build_app(args.config)
    handler = COMMANDS[args.command]
    return handler(app, args)


if __name__ == "__main__":
    sys.exit(main())
