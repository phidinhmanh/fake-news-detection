@echo off
:: verity.bat — Windows wrapper for Verity CLI
:: Usage: verity.bat <command> [options]
:: Or:    verity <command> [options]  (if added to PATH)

setlocal

:: Get script directory
set "SCRIPT_DIR=%~dp0"

:: Run with uv
uv run python "%SCRIPT_DIR%verity.py" %*

endlocal