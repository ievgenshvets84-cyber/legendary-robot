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
import io
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
                 sampler: str = "DPM++ 2M Karras", upscaler: str = "R-ESRGAN 4x+",
                 upscale: float = 2.0, timeout: int = 600) -> None:
        self.guard = guard
        self.save_dir = save_dir
        self.image_host = image_host.rstrip("/")
        self.video_host = video_host.rstrip("/")
        self.steps = steps
        self.width = width
        self.height = height
        self.cfg_scale = cfg_scale
        self.sampler = sampler
        self.upscaler = upscaler
        self.upscale = upscale
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

    # ── image-to-image ───────────────────────────────────────────────────
    def image_to_image(self, prompt: str, init_image: str,
                       negative_prompt: str = "", denoising_strength: float = 0.6,
                       width: int | None = None, height: int | None = None,
                       steps: int | None = None, seed: int = -1,
                       count: int = 1) -> list[Path]:
        """Перерисовать существующее изображение по текстовому запросу.

        denoising_strength: 0.0 — почти без изменений, 1.0 — полностью заново.
        """
        if not self.image_host:
            raise MediaError("Не задан media.image_host в config.yaml.")
        payload = {
            "init_images": [self._read_image_b64(init_image)],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": denoising_strength,
            "steps": steps or self.steps,
            "width": width or self.width,
            "height": height or self.height,
            "cfg_scale": self.cfg_scale,
            "sampler_name": self.sampler,
            "batch_size": max(1, count),
            "seed": seed,
        }
        data = self._post(self.image_host + "/sdapi/v1/img2img", payload)
        images = data.get("images") or []
        if not images:
            raise MediaError("Сервер изображений не вернул данных.")
        return self._save_many(images, prefix="i2i", ext="png")

    # ── inpainting ───────────────────────────────────────────────────────
    def inpaint(self, prompt: str, init_image: str, mask_image: str,
                negative_prompt: str = "", denoising_strength: float = 0.75,
                mask_blur: int = 4, inpainting_fill: int = 1,
                inpaint_full_res: bool = True, steps: int | None = None,
                seed: int = -1) -> list[Path]:
        """Перерисовать только область под маской (белое = править, чёрное = оставить).

        inpainting_fill: 0=fill, 1=original, 2=latent noise, 3=latent nothing.
        """
        if not self.image_host:
            raise MediaError("Не задан media.image_host в config.yaml.")
        payload = {
            "init_images": [self._read_image_b64(init_image)],
            "mask": self._read_image_b64(mask_image),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": denoising_strength,
            "mask_blur": mask_blur,
            "inpainting_fill": inpainting_fill,
            "inpaint_full_res": inpaint_full_res,
            "steps": steps or self.steps,
            "cfg_scale": self.cfg_scale,
            "sampler_name": self.sampler,
            "seed": seed,
        }
        data = self._post(self.image_host + "/sdapi/v1/img2img", payload)
        images = data.get("images") or []
        if not images:
            raise MediaError("Сервер изображений не вернул данных.")
        return self._save_many(images, prefix="inpaint", ext="png")

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
        return self._handle_video_response(data, fps)

    # ── image-to-video (анимация одной картинки, Stable Video Diffusion) ──
    def image_to_video(self, init_image: str, prompt: str = "",
                       negative_prompt: str = "", seconds: float = 2.0,
                       fps: int = 8, motion: int = 127) -> list[Path]:
        """Оживить одно изображение в короткий ролик через локальный SVD-бэкенд.

        motion — «сила движения» (motion_bucket_id в SVD): больше = динамичнее.
        """
        if not self.video_host:
            raise MediaError(
                "Локальный видео-бэкенд не настроен (media.video_host пуст). "
                "Поднимите локальный сервер (обёртку над ComfyUI со Stable Video "
                "Diffusion) с эндпоинтом /img2video и укажите его адрес."
            )
        payload = {
            "init_image": self._read_image_b64(init_image),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": max(1, int(seconds * fps)),
            "fps": fps,
            "motion_bucket_id": motion,
        }
        data = self._post(self.video_host + "/img2video", payload)
        return self._handle_video_response(data, fps)

    # ── сборка клипа из нескольких готовых картинок ───────────────────────
    def images_to_video(self, images: list[str], fps: int = 8) -> list[Path]:
        """Собрать ролик из последовательности сгенерированных изображений.

        Работает локально без видео-бэкенда: кадры склеиваются в анимированный
        GIF (нужен Pillow). Порядок кадров — как передан в списке.
        """
        if not images:
            raise MediaError("Не переданы изображения для сборки видео.")
        frames: list[Path] = []
        for path in images:
            target = self.guard.resolve_path(path)
            if not target.exists():
                raise MediaError(f"Кадр не найден: {path}")
            frames.append(target)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = self._save_dir() / f"clip-{stamp}.gif"
        gif = self._assemble_gif(frames, fps=fps, out_path=out)
        if not gif:
            raise MediaError("Для сборки видео из картинок нужен Pillow: "
                             "pip install Pillow")
        return [gif]

    # ── апскейл изображений и видео ──────────────────────────────────────
    def upscale_image(self, image: str, scale: float | None = None,
                      upscaler: str | None = None) -> list[Path]:
        """Увеличить разрешение изображения апскейлером SD-сервера (ESRGAN и т.п.)."""
        if not self.image_host:
            raise MediaError("Не задан media.image_host в config.yaml.")
        b64 = self._upscale_b64(self._read_image_b64(image),
                                scale or self.upscale, upscaler or self.upscaler)
        return self._save_many([b64], prefix="up", ext="png")

    def upscale_video(self, video: str, scale: float | None = None,
                      upscaler: str | None = None, fps: int = 8) -> list[Path]:
        """Покадрово увеличить разрешение GIF-клипа и пересобрать его.

        Каждый кадр апскейлится тем же сервером, что и картинки. Для mp4 нужен
        ffmpeg — такой ролик сначала разложите на кадры/GIF.
        """
        if not self.image_host:
            raise MediaError("Для апскейла кадров нужен media.image_host.")
        target = self.guard.resolve_path(video)
        if not target.exists():
            raise MediaError(f"Файл видео не найден: {video}")
        if target.suffix.lower() != ".gif":
            raise MediaError(
                "Апскейл видео поддержан для GIF-клипов (slideshow/img2video без "
                "бэкенда). Для mp4 нужен ffmpeg — разложите ролик на кадры."
            )
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            raise MediaError("Для апскейла видео нужен Pillow: pip install Pillow")

        factor = scale or self.upscale
        up = upscaler or self.upscaler
        clip = Image.open(target)
        upscaled_b64: list[str] = []
        for i in range(getattr(clip, "n_frames", 1)):
            clip.seek(i)
            buf = io.BytesIO()
            clip.convert("RGB").save(buf, format="PNG")
            frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            upscaled_b64.append(self._upscale_b64(frame_b64, factor, up))
        saved = self._save_many(upscaled_b64, prefix="upframe", ext="png")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = self._save_dir() / f"clip-up-{stamp}.gif"
        gif = self._assemble_gif(saved, fps=fps, out_path=out)
        return [gif] if gif else saved

    def _upscale_b64(self, image_b64: str, scale: float, upscaler: str) -> str:
        payload = {
            "resize_mode": 0,               # 0 = увеличить в N раз
            "upscaling_resize": scale,
            "upscaler_1": upscaler,
            "image": image_b64,
        }
        data = self._post(self.image_host + "/sdapi/v1/extra-single-image", payload)
        out = data.get("image")
        if not out:
            raise MediaError("Апскейлер не вернул изображение.")
        return out

    def _handle_video_response(self, data: dict[str, Any], fps: int) -> list[Path]:
        """Разбирает ответ видео-сервера: готовый ролик или набор кадров."""
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

    def _read_image_b64(self, path: str) -> str:
        """Читает изображение из песочницы и кодирует в base64 для API."""
        target = self.guard.resolve_path(path)
        if not target.exists():
            raise MediaError(f"Файл изображения не найден: {path}")
        return base64.b64encode(target.read_bytes()).decode("ascii")

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

    def _assemble_gif(self, frames: list[Path], fps: int,
                      out_path: Path | None = None) -> Path | None:
        """Собирает кадры в анимированный GIF, если доступен Pillow."""
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return None
        if not frames:
            return None
        imgs = [Image.open(p).convert("RGB") for p in frames]
        gif_path = out_path or frames[0].with_name(
            frames[0].stem.replace("frame", "clip") + ".gif")
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
            is_image = "sdapi" in url
            what = ("Stable Diffusion для картинок" if is_image
                    else "видео-бэкенд")
            fix = ("запустите Automatic1111/SD.Next с флагом --api (порт 7860)"
                   if is_image else "запустите локальный видео-сервер (ComfyUI/SVD)")
            raise MediaError(
                f"Нет связи с сервером генерации по адресу {url}. "
                f"Это ОТДЕЛЬНЫЙ локальный сервер ({what}), а НЕ Ollama — "
                f"он не запускается автоматически. Чтобы генерировать медиа, {fix}; "
                f"подробности в разделе «Медиа» в README. Причина: {exc}"
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
    def _img2img(ctx, prompt: str, init_image: str, negative_prompt: str = "",
                 strength: str = "0.6") -> str:
        try:
            denoise = max(0.0, min(1.0, float(strength)))
        except (TypeError, ValueError):
            denoise = 0.6
        paths = media.image_to_image(prompt, init_image=init_image,
                                     negative_prompt=negative_prompt,
                                     denoising_strength=denoise)
        return "Перерисованные изображения:\n" + "\n".join(str(p) for p in paths)

    def _inpaint(ctx, prompt: str, init_image: str, mask_image: str,
                 negative_prompt: str = "", strength: str = "0.75") -> str:
        try:
            denoise = max(0.0, min(1.0, float(strength)))
        except (TypeError, ValueError):
            denoise = 0.75
        paths = media.inpaint(prompt, init_image=init_image, mask_image=mask_image,
                              negative_prompt=negative_prompt,
                              denoising_strength=denoise)
        return "Результат inpainting:\n" + "\n".join(str(p) for p in paths)

    registry.register(Tool(
        "image_to_image",
        "Перерисовать существующее изображение по текстовому запросу "
        "(img2img на локальной модели).",
        {"prompt": "как изменить/что нарисовать",
         "init_image": "путь к исходному изображению в рабочей папке",
         "negative_prompt": "чего избегать (необязательно)",
         "strength": "сила изменений 0.0–1.0 (необязательно)"},
        _img2img,
    ))
    registry.register(Tool(
        "inpaint_image",
        "Перерисовать только область под маской (inpainting): белое в маске — "
        "править, чёрное — оставить.",
        {"prompt": "что должно появиться в области",
         "init_image": "путь к исходному изображению",
         "mask_image": "путь к чёрно-белой маске",
         "negative_prompt": "чего избегать (необязательно)",
         "strength": "сила изменений 0.0–1.0 (необязательно)"},
        _inpaint,
    ))
    def _img2video(ctx, init_image: str, prompt: str = "", seconds: str = "2") -> str:
        try:
            secs = float(seconds)
        except (TypeError, ValueError):
            secs = 2.0
        paths = media.image_to_video(init_image=init_image, prompt=prompt, seconds=secs)
        return "Оживлённое изображение (видео/кадры):\n" + "\n".join(str(p) for p in paths)

    def _images2video(ctx, images: str, fps: str = "8") -> str:
        items = [s.strip() for s in images.split(",") if s.strip()]
        try:
            rate = max(1, int(fps))
        except (TypeError, ValueError):
            rate = 8
        paths = media.images_to_video(items, fps=rate)
        return "Клип из картинок:\n" + "\n".join(str(p) for p in paths)

    registry.register(Tool(
        "generate_video",
        "Сгенерировать короткое видео локальным видео-бэкендом и сохранить "
        "в рабочую папку.",
        {"prompt": "описание сцены",
         "negative_prompt": "чего избегать (необязательно)",
         "seconds": "длительность в секундах (необязательно)"},
        _gen_video,
    ))
    def _upscale_image(ctx, image: str, scale: str = "2") -> str:
        try:
            factor = float(scale)
        except (TypeError, ValueError):
            factor = 2.0
        paths = media.upscale_image(image, scale=factor)
        return "Увеличенное изображение:\n" + "\n".join(str(p) for p in paths)

    def _upscale_video(ctx, video: str, scale: str = "2") -> str:
        try:
            factor = float(scale)
        except (TypeError, ValueError):
            factor = 2.0
        paths = media.upscale_video(video, scale=factor)
        return "Увеличенный клип:\n" + "\n".join(str(p) for p in paths)

    registry.register(Tool(
        "upscale_image",
        "Увеличить разрешение изображения (ESRGAN/R-ESRGAN на локальном SD-сервере).",
        {"image": "путь к изображению в рабочей папке",
         "scale": "кратность увеличения, напр. 2 или 4 (необязательно)"},
        _upscale_image,
    ))
    registry.register(Tool(
        "upscale_video",
        "Покадрово увеличить разрешение GIF-клипа и пересобрать его.",
        {"video": "путь к GIF-клипу в рабочей папке",
         "scale": "кратность увеличения (необязательно)"},
        _upscale_video,
    ))
    registry.register(Tool(
        "image_to_video",
        "Оживить одно сгенерированное изображение в короткий ролик "
        "(image-to-video через локальный Stable Video Diffusion).",
        {"init_image": "путь к исходному изображению в рабочей папке",
         "prompt": "подсказка по движению (необязательно)",
         "seconds": "длительность в секундах (необязательно)"},
        _img2video,
    ))
    registry.register(Tool(
        "images_to_video",
        "Собрать короткий клип из последовательности сгенерированных картинок.",
        {"images": "пути к изображениям через запятую, в нужном порядке",
         "fps": "кадров в секунду (необязательно)"},
        _images2video,
    ))
