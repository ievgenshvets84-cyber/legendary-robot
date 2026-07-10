"""Локальная генерация медиа: изображения и видео на открытых моделях.

Изображения генерируются через локальный сервер, совместимый с API
AUTOMATIC1111 / SD.Next (эндпоинт /sdapi/v1/txt2img) — то есть Stable
Diffusion / SDXL / любые открытые чекпойнты, запущенные у вас на машине.
Видео — через локальный видео-бэкенд (обёртку над ComfyUI / Stable Video
Diffusion и т.п.), либо покадрово из изображений.

Как и всё в LocalMind, работает офлайн: данные не покидают вашу машину.
Модуль не отключает и не обходит защитные механизмы генеративных моделей —
он лишь обращается к тому локальному движку, который настроил пользователь.
Ответственность за законность создаваемого контента лежит на пользователе.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .safety import Guard
from .tools import Tool, ToolRegistry


class MediaError(RuntimeError):
    pass


class MediaClient:
    def __init__(self, guard: Guard, save_dir: str = "state/media",
                 image_host: str = "http://localhost:7860",
                 video_host: str = "", steps: int = 30,
                 width: int = 768, height: int = 768, cfg_scale: float = 7.0,
                 sampler: str = "DPM++ 2M Karras", timeout: int = 600) -> None:
        self.guard = guard
        self.save_dir = save_dir
        self.image_host = image_host.rstrip("/")
        self.video_host = video_host.rstrip("/")
        self.steps = steps
        self.width = width
        self.height = height
        self.cfg_scale = cfg_scale
        self.sampler = sampler
        self.timeout = timeout

    # ── изображения ──────────────────────────────────────────────────────
    def generate_image(self, prompt: str, negative_prompt: str = "",
                        width: int | None = None, height: int | None = None,
                        steps: int | None = None, seed: int = -1,
                        count: int = 1) -> list[Path]:
        if not self.image_host:
            raise MediaError("Не задан media.image_host в config.yaml.")
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps or self.steps,
            "width": width or self.width,
            "height": height or self.height,
            "cfg_scale": self.cfg_scale,
            "sampler_name": self.sampler,
            "batch_size": max(1, count),
            "seed": seed,
        }
        data = self._post(self.image_host + "/sdapi/v1/txt2img", payload)
        images = data.get("images") or []
        if not images:
            raise MediaError("Сервер изображений не вернул данных.")
        return self._save_many(images, prefix="img", ext="png")

    # ── видео ────────────────────────────────────────────────────────────
    def generate_video(self, prompt: str, negative_prompt: str = "",
                        seconds: float = 2.0, fps: int = 8) -> list[Path]:
        if not self.video_host:
            raise MediaError(
                "Локальный видео-бэкенд не настроен (media.video_host пуст). "
                "Поднимите локальный сервер видео (например, обёртку над ComfyUI "
                "со Stable Video Diffusion / AnimateDiff) и укажите его адрес."
            )
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": max(1, int(seconds * fps)),
            "fps": fps,
        }
        data = self._post(self.video_host + "/generate", payload)
        # Поддерживаем два формата ответа: готовый ролик или набор кадров.
        if data.get("video"):
            return self._save_many([data["video"]], prefix="vid", ext="mp4")
        frames = data.get("frames") or []
        if not frames:
            raise MediaError("Видео-сервер не вернул ни ролика, ни кадров.")
        saved = self._save_many(frames, prefix="frame", ext="png")
        gif = self._assemble_gif(saved, fps=fps)
        return [gif] if gif else saved

    # ── доступность ──────────────────────────────────────────────────────
    def image_available(self) -> bool:
        return self._ping(self.image_host + "/sdapi/v1/sd-models") if self.image_host else False

    def video_available(self) -> bool:
        return self._ping(self.video_host + "/health") if self.video_host else False

    # ── служебное ────────────────────────────────────────────────────────
    def _save_dir(self) -> Path:
        target = self.guard.resolve_path(self.save_dir)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _save_many(self, b64_items: list[str], prefix: str, ext: str) -> list[Path]:
        out_dir = self._save_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        paths: list[Path] = []
        for i, item in enumerate(b64_items):
            raw = item.split(",", 1)[-1]  # отсечь возможный data:URI-префикс
            blob = base64.b64decode(raw)
            path = out_dir / f"{prefix}-{stamp}-{i}.{ext}"
            path.write_bytes(blob)
            paths.append(path)
        return paths

    def _assemble_gif(self, frames: list[Path], fps: int) -> Path | None:
        """Собирает кадры в анимированный GIF, если доступен Pillow."""
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return None
        if not frames:
            return None
        imgs = [Image.open(p) for p in frames]
        gif_path = frames[0].with_name(frames[0].stem.replace("frame", "clip") + ".gif")
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(1, fps)), loop=0)
        return gif_path

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise MediaError(
                f"Не удаётся связаться с локальным медиа-сервером ({url}). "
                f"Убедитесь, что он запущен. Причина: {exc}"
            ) from exc

    def _ping(self, url: str) -> bool:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False


# ── регистрация инструментов агента ──────────────────────────────────────
def register_media_tools(registry: ToolRegistry, media: MediaClient) -> None:
    def _gen_image(ctx, prompt: str, negative_prompt: str = "", count: str = "1") -> str:
        try:
            n = max(1, min(4, int(count)))
        except (TypeError, ValueError):
            n = 1
        paths = media.generate_image(prompt, negative_prompt=negative_prompt, count=n)
        return "Сохранены изображения:\n" + "\n".join(str(p) for p in paths)

    def _gen_video(ctx, prompt: str, negative_prompt: str = "", seconds: str = "2") -> str:
        try:
            secs = float(seconds)
        except (TypeError, ValueError):
            secs = 2.0
        paths = media.generate_video(prompt, negative_prompt=negative_prompt, seconds=secs)
        return "Сохранено видео/кадры:\n" + "\n".join(str(p) for p in paths)

    registry.register(Tool(
        "generate_image",
        "Сгенерировать изображение локальной моделью (Stable Diffusion / SDXL) "
        "и сохранить в рабочую папку.",
        {"prompt": "описание сцены",
         "negative_prompt": "чего избегать (необязательно)",
         "count": "сколько картинок 1–4 (необязательно)"},
        _gen_image,
    ))
    registry.register(Tool(
        "generate_video",
        "Сгенерировать короткое видео локальным видео-бэкендом и сохранить "
        "в рабочую папку.",
        {"prompt": "описание сцены",
         "negative_prompt": "чего избегать (необязательно)",
         "seconds": "длительность в секундах (необязательно)"},
        _gen_video,
    ))
