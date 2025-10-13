# 🚀 Complete GitHub Integration Guide for Cursor

## 📋 **Prerequisites**
- Git installed on your system
- GitHub account
- Cursor IDE

## 🔧 **Step 1: Install Git**

### **Download Git:**
1. Go to: https://git-scm.com/download/win
2. Download the latest version
3. Run installer with **default settings**
4. **Important**: Select "Git from the command line and also from 3rd-party software"

### **Verify Installation:**
```bash
git --version
```

## ⚙️ **Step 2: Configure Git**

### **Set your identity:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### **Set default branch name:**
```bash
git config --global init.defaultBranch main
```

## 🔐 **Step 3: Set up SSH Key (Recommended)**

### **Generate SSH key:**
```bash
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
```
- Press Enter for default file location
- Enter a passphrase (optional but recommended)

### **Add SSH key to GitHub:**
1. Copy your public key:
   ```bash
   type %USERPROFILE%\.ssh\id_rsa.pub
   ```
2. Go to GitHub.com → Settings → SSH and GPG keys
3. Click "New SSH key"
4. Paste the key and save

### **Test SSH connection:**
```bash
ssh -T git@github.com
```

## 📁 **Step 4: Initialize Repository in Cursor**

### **Initialize Git repository:**
```bash
git init
```

### **Create .gitignore file:**
```bash
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
```

### **Add files to repository:**
```bash
git add .
git commit -m "Initial commit: CTVO multi-view virtual try-on project"
```

## 🌐 **Step 5: Create GitHub Repository**

### **On GitHub.com:**
1. Click "New repository"
2. Repository name: `CTVO-MultiView-VirtualTryOn`
3. Description: `Multi-view virtual try-on system with NeRF integration`
4. Set to **Public** or **Private**
5. **Don't** initialize with README (we already have files)
6. Click "Create repository"

## 🔗 **Step 6: Connect Local Repository to GitHub**

### **Add remote origin:**
```bash
git remote add origin git@github.com:YOUR_USERNAME/CTVO-MultiView-VirtualTryOn.git
```

### **Push to GitHub:**
```bash
git branch -M main
git push -u origin main
```

## 🎯 **Step 7: Using Git in Cursor**

### **Cursor Git Features:**
1. **Source Control Panel** (Ctrl+Shift+G)
2. **Git Status** in status bar
3. **Commit changes** directly in Cursor
4. **Push/Pull** from command palette
5. **Branch management** in bottom status bar

### **Common Git Commands in Cursor:**
- **Commit**: Ctrl+Shift+G → Stage changes → Commit
- **Push**: Ctrl+Shift+P → "Git: Push"
- **Pull**: Ctrl+Shift+P → "Git: Pull"
- **Create Branch**: Ctrl+Shift+P → "Git: Create Branch"

## 📝 **Step 8: Project-Specific Git Workflow**

### **For CTVO Project:**
```bash
# Add new features
git add .
git commit -m "Add Stage 3 fusion generator"

# Push to GitHub
git push origin main

# Create feature branch
git checkout -b feature/stage4-nerf
git add .
git commit -m "Implement Stage 4 NeRF rendering"
git push origin feature/stage4-nerf
```

## 🔄 **Step 9: Collaboration Features**

### **Pull Requests:**
1. Create feature branch
2. Make changes
3. Push branch
4. Create Pull Request on GitHub
5. Review and merge

### **Issues:**
1. Go to GitHub repository
2. Click "Issues" tab
3. Create new issue
4. Assign labels and milestones

## 🛡️ **Step 10: Security Best Practices**

### **Never commit:**
- API keys
- Passwords
- Personal data
- Large model files
- Sensitive configuration

### **Use .gitignore:**
- Add sensitive files to .gitignore
- Use environment variables for secrets
- Keep configuration files separate

## 🎉 **Success Checklist**

✅ Git installed and configured  
✅ SSH key generated and added to GitHub  
✅ Repository initialized  
✅ .gitignore created  
✅ GitHub repository created  
✅ Local repository connected to GitHub  
✅ First commit pushed  
✅ Cursor Git integration working  

## 🆘 **Troubleshooting**

### **Git not found:**
- Restart Cursor after Git installation
- Check PATH environment variable
- Use full path: `C:\Program Files\Git\bin\git.exe`

### **SSH connection failed:**
- Check SSH key is added to GitHub
- Test with: `ssh -T git@github.com`
- Regenerate SSH key if needed

### **Push rejected:**
- Pull first: `git pull origin main`
- Resolve conflicts
- Push again: `git push origin main`

---

**Ready to start? Install Git first, then follow the steps above!** 🚀
