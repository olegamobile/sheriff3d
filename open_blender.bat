@echo off
chcp 65001 > nul
echo ========================================================
echo   Запуск Blender 5.2 с лодкой Jouët Sheriff 600
echo   Автоматический запуск Blender MCP сервера (порт 9876)
echo ========================================================
start "" "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" "%~dp0Jouet_Sheriff_600.blend" --python "%~dp0auto_start_mcp.py"
