# Setup Guide - Virtual Environment and Installation

This guide will help you set up a virtual environment and install all required dependencies for the trash detection project.

## Prerequisites

- Python 3.8 or higher installed on your system
- pip (Python package installer)

## Step 1: Create a Virtual Environment

### On Windows (PowerShell):
```powershell
# Navigate to the project directory
cd c:\Users\suson\Downloads\trash_detection

# Create virtual environment
python -m venv venv

# OR if python doesn't work, try:
py -m venv venv
# OR
python3 -m venv venv
```

### On Windows (Command Prompt):
```cmd
cd c:\Users\suson\Downloads\trash_detection
python -m venv venv
```

### On Linux/Mac:
```bash
cd ~/Downloads/trash_detection
python3 -m venv venv
```

## Step 2: Activate the Virtual Environment

### On Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### On Windows (Command Prompt):
```cmd
venv\Scripts\activate.bat
```

### On Linux/Mac:
```bash
source venv/bin/activate
```

After activation, you should see `(venv)` at the beginning of your command prompt.

## Step 3: Install Requirements

Once the virtual environment is activated, install all dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Verify Installation

You can verify the installation by checking if packages are installed:

```bash
pip list
```

## Step 5: Deactivate (when done)

When you're finished working, you can deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Python not found
- Make sure Python is installed and added to your PATH
- Download Python from https://www.python.org/downloads/
- During installation, check "Add Python to PATH"

### Permission errors
- Make sure you have write permissions in the project directory
- Try running PowerShell/Command Prompt as Administrator

### Virtual environment activation issues
- Make sure you're in the correct directory
- On Windows PowerShell, you may need to allow script execution (see Step 2)

### Installation errors
- Some packages (like `tflite-runtime`) may not be available for Windows
- On Windows, you may need to use `tensorflow` instead of `tflite-runtime`
- Edit `requirements.txt` and uncomment the `tensorflow` line if needed

## Quick Start (All-in-One Commands)

### Windows PowerShell:
```powershell
cd c:\Users\suson\Downloads\trash_detection
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Command Prompt:
```cmd
cd c:\Users\suson\Downloads\trash_detection
python -m venv venv
venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

