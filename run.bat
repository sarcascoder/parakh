@echo off
REM One-click: run the offline demo, then launch the review UI (Windows).
cd /d "%~dp0"
echo ==^> Plumb demo (offline, no model needed)
set PYTHONPATH=.
python -m examples.invoices.run_demo
if /I "%1"=="demo" goto :eof
echo.
echo ==^> Starting review UI at http://127.0.0.1:8000  (press Ctrl-C to stop)
python -m plumb.cli review
