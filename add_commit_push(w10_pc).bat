@echo off
set datetime=%DATE% %TIME%
set datetime=%datetime:,=:%

set commit_msg=W10 PC %datetime%

git pull

git add .
git commit -m "%commit_msg%"
git push origin main

rem pause