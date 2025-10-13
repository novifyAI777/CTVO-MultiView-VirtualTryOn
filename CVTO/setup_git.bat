@echo off
REM Quick Git Setup Script
REM Run this AFTER installing Git

echo ========================================
echo Quick Git Setup for CTVO Project
echo ========================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git not found!
    echo Please install Git first from: https://git-scm.com/download/win
    echo.
    echo After installing Git:
    echo 1. Restart Cursor
    echo 2. Run this script again
    pause
    exit /b 1
)

echo ✅ Git found!
git --version
echo.

echo Setting up Git configuration...
echo.
echo Please enter your information:
echo.

set /p USER_NAME="Enter your name: "
set /p USER_EMAIL="Enter your email: "

echo.
echo Configuring Git...
git config --global user.name "%USER_NAME%"
git config --global user.email "%USER_EMAIL%"
git config --global init.defaultBranch main

echo.
echo ✅ Git configured!
echo.

echo Initializing repository...
git init

echo.
echo Creating .gitignore file...
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo build/
echo develop-eggs/
echo dist/
echo downloads/
echo eggs/
echo .eggs/
echo lib/
echo lib64/
echo parts/
echo sdist/
echo var/
echo wheels/
echo *.egg-info/
echo .installed.cfg
echo *.egg
echo.
echo # PyTorch
echo *.pth
echo *.pt
echo *.pkl
echo.
echo # Data
echo data/raw/
echo data/processed/
echo results/
echo logs/
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
echo.
echo # Temporary files
echo temp_organization/
echo *.tmp
echo *.log
) > .gitignore

echo.
echo Adding files to repository...
git add .

echo.
echo Making initial commit...
git commit -m "Initial commit: CTVO multi-view virtual try-on project"

echo.
echo ========================================
echo ✅ GIT REPOSITORY INITIALIZED!
echo ========================================
echo.
echo Next steps:
echo 1. Create GitHub repository at: https://github.com/new
echo 2. Repository name: CTVO-MultiView-VirtualTryOn
echo 3. Don't initialize with README
echo 4. Copy the repository URL
echo 5. Run: git remote add origin YOUR_REPO_URL
echo 6. Run: git push -u origin main
echo.
echo For detailed instructions, see: GITHUB_INTEGRATION_GUIDE.md
echo.

pause
