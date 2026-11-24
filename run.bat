@echo off
REM ============================================================
REM  CandyCaption launcher - double-click to run (no terminal needed).
REM
REM  Needs uv (https://docs.astral.sh/uv/). The first launch installs
REM  the dependencies by itself; nothing here loads a model.
REM ============================================================
cd /d "%~dp0"

echo ============================================================
echo   CandyCaption is starting up...
echo.
echo   Web interface:  http://localhost:8200
echo.
echo   Start your model server first ^(LM Studio, Unsloth Studio,
echo   llama.cpp or vLLM^), then pick it in the Candy Shop tab.
echo   Close this window to stop CandyCaption.
echo ============================================================
echo.

REM A hidden background PowerShell polls /health and opens the browser only
REM once the server answers, so the page never opens too early.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 150;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://localhost:8200/health').StatusCode -eq 200){Start-Process 'http://localhost:8200';exit}}catch{}; Start-Sleep -Seconds 2}"

uv run uvicorn candycaption.server:app --host 127.0.0.1 --port 8200

echo.
echo CandyCaption has stopped.
pause
