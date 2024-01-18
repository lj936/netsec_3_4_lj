@echo off
set datetime=%DATE% %TIME%
set datetime=%datetime:,=:%

set commit_msg=W10 PC %datetime%

git pull

rem Füge nur Dateien hinzu, die keine JSON-Dateien sind
for %%i in (*) do (
    if /I "%%~xi" neq ".json" (
        git add "%%i"
    )
)

rem git add .
git commit -m "%commit_msg%"
git push origin main

rem pause