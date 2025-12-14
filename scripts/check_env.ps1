Write-Host "=== Environment Check: python and pytest ===`n"

# Check for python (also check py launcher)
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue

if ($pythonCmd) {
    Write-Host "Found 'python' in PATH: $($pythonCmd.Source)"
    try {
        & python --version 2>&1 | Write-Host
    } catch { Write-Host "Unable to run 'python --version'" }
} elseif ($pyLauncher) {
    Write-Host "Found 'py' launcher: $($pyLauncher.Source)"
    try {
        & py -3 --version 2>&1 | Write-Host
    } catch { Write-Host "Unable to run 'py -3 --version'" }
} else {
    Write-Host "Python not found in PATH."
}

Write-Host `n"Checking for a virtual environment at .venv..."
if (Test-Path -Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Virtual environment found at .venv"
} else {
    Write-Host "No local virtual environment (.venv) detected"
}

Write-Host `n"Checking pytest availability..."
$pytestCmd = Get-Command pytest -ErrorAction SilentlyContinue
if ($pytestCmd) {
    Write-Host "Found 'pytest' on PATH: $($pytestCmd.Source)"
    try { & pytest --version 2>&1 | Write-Host } catch { Write-Host "Unable to run 'pytest --version'" }
} else {
    # Try python -m pytest if python exists
    if ($pythonCmd -or $pyLauncher) {
        try {
            if ($pythonCmd) {
                & python -m pytest --version 2>&1 | Write-Host
            } else {
                & py -3 -m pytest --version 2>&1 | Write-Host
            }
        } catch {
            Write-Host "pytest not available as a module. Install with: python -m pip install pytest"
        }
    } else {
        Write-Host "pytest could not be checked because Python is not available."
    }
}

Write-Host `n"Recommendations:";
if (-not ($pythonCmd -or $pyLauncher)) {
    Write-Host "- Install Python 3.8+ and ensure it's added to PATH."
    Write-Host "  Download: https://www.python.org/downloads/"
    Write-Host "- After installing, re-open PowerShell and run: python --version"
} else {
    Write-Host "- If you want to run tests, create and activate a virtualenv then install requirements:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .\\.venv\\Scripts\\Activate.ps1"
    Write-Host "  python -m pip install --upgrade pip"
    Write-Host "  pip install -r requirements.txt"
    Write-Host "  python -m pytest tests/ -v"
}

Write-Host `n"=== End Environment Check ==="