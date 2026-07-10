#!/usr/bin/env bash
# Запуск LocalMind в один шаг (Linux / macOS): поднимает Ollama и веб-интерфейс.
set -e

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Ошибка] Python 3 не найден. Установите с https://python.org"; exit 1
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "[Ошибка] Ollama не найден. Установите с https://ollama.com/download"; exit 1
fi

# Запустить сервер Ollama в фоне, если он ещё не отвечает.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Запускаю сервер Ollama…"
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi

cd "$(dirname "$0")"
echo "Запускаю веб-интерфейс LocalMind… (Ctrl+C — стоп)"
echo "Примечание: чат и агент работают сразу. Для генерации картинок (режим «Медиа»)"
echo "  нужен отдельный сервер Stable Diffusion на порту 7860 (Automatic1111 --api)."
exec python3 -m localmind web
