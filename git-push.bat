@echo off
echo ==========================================
echo Pushing Movie Recommender to GitHub...
echo ==========================================
cd /d "%~dp0"

echo 1. Staging all files...
git add .

echo 2. Committing changes...
git commit -m "Update project files and add custom port launch configuration"

echo 3. Pushing to GitHub (main branch)...
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Git push failed. Attempting force push...
    git push -u origin main --force
)

echo ==========================================
echo Done! Please check the terminal output above.
echo ==========================================
pause
