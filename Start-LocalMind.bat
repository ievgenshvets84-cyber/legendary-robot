@echo off
chcp 65001 >nul
title LocalMind
echo ============================================
echo   LocalMind - Start / Запуск
echo ============================================
echo.

REM --- Python pruefen / проверка Python ---
where python >nul 2>nul
if errorlevel 1 (
  echo [Fehler] Python nicht gefunden. Bitte von https://python.org installieren.
  echo [Ошибка] Python не найден. Установите с https://python.org
  pause
  exit /b 1
)

REM --- Ollama pruefen / проверка Ollama ---
where ollama >nul 2>nul
if errorlevel 1 (
  echo [Fehler] Ollama nicht gefunden. Bitte von https://ollama.com/download installieren.
  echo [Ошибка] Ollama не найден. Установите с https://ollama.com/download
  pause
  exit /b 1
)

REM --- Ollama-Server im Hintergrund starten / запустить сервер Ollama ---
echo Starte Ollama-Server... / Запускаю сервер Ollama...
start "Ollama" /min ollama serve

REM --- kurz warten / короткая пауза, пока сервер поднимется ---
timeout /t 3 >nul

REM --- Web-Oberflaeche starten (oeffnet Browser) / веб-интерфейс (откроет браузер) ---
echo Starte LocalMind Web-UI... / Запускаю веб-интерфейс LocalMind...
echo Zum Beenden dieses Fenster schliessen. / Чтобы остановить - закройте это окно.
echo.
python -m localmind web

pause
