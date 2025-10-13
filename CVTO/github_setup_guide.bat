@echo off
REM Git Setup and GitHub Connection Script
REM Run this AFTER installing Git

echo ========================================
echo Git Setup and GitHub Connection
echo ========================================
echo.

echo Step 1: Install Git (if not already installed)
echo Download from: https://git-scm.com/download/win
echo.
echo Step 2: Configure Git with your information
echo.
echo After installing Git, run these commands:
echo.
echo git config --global user.name "Your Name"
echo git config --global user.email "your.email@example.com"
echo.
echo Step 3: Generate SSH Key (for secure connection)
echo.
echo ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
echo.
echo Step 4: Add SSH Key to GitHub
echo 1. Copy the public key: type %USERPROFILE%\.ssh\id_rsa.pub
echo 2. Go to GitHub.com → Settings → SSH and GPG keys
echo 3. Click "New SSH key"
echo 4. Paste the key and save
echo.
echo Step 5: Test connection
echo ssh -T git@github.com
echo.
echo Step 6: Initialize repository in Cursor
echo git init
echo git add .
echo git commit -m "Initial commit"
echo git branch -M main
echo git remote add origin git@github.com:username/repository.git
echo git push -u origin main
echo.

pause
