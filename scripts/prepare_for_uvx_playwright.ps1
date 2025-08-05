# UVX Playwright System Preparation Script (Windows PowerShell)
# Purpose: Prepare system dependencies for UVX crawl4ai-dxt-correct execution
# Usage: Run as Administrator - powershell -ExecutionPolicy Bypass -File scripts/prepare_for_uvx_playwright.ps1

param()

$ErrorActionPreference = 'Stop'

# Check for administrator privileges
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "[ERROR] Please run PowerShell as Administrator" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Get Playwright version from requirements.txt
function Get-PlaywrightVersionFromRequirements {
    param([string[]]$RequirementsPaths)
    
    foreach ($path in $RequirementsPaths) {
        if (Test-Path $path) {
            $content = Get-Content $path -ErrorAction SilentlyContinue
            foreach ($line in $content) {
                if ($line -match '^playwright==(.+)$') {
                    return "playwright==$($matches[1])"
                }
                if ($line -match '^playwright>=(.+)$') {
                    return "playwright>=$($matches[1])"
                }
            }
        }
    }
    
    # Default fallback
    return 'playwright==1.54.0'
}

# Check Python installation
function Test-PythonEnvironment {
    try {
        $pythonVersion = & python --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw 'Python not found'
        }
        
        # Test venv module
        & python -m venv --help >$null 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw 'venv module not available' 
        }
        
        # Test pip
        & python -m pip --version >$null 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw 'pip not available'
        }
        
        return $true
    } catch {
        Write-Host '[ERROR] Python environment issues detected:' -ForegroundColor Red
        Write-Host "  - Python 3.7+ required (current: $(if($pythonVersion) { $pythonVersion } else { 'not installed' }))" -ForegroundColor Yellow
        Write-Host '  - Please install Python from:' -ForegroundColor Yellow
        Write-Host '    https://www.python.org/downloads/windows/' -ForegroundColor Yellow
        Write-Host '  - Make sure to check "Add Python to PATH" during installation' -ForegroundColor Yellow
        return $false
    }
}

# Message functions
function Write-InfoMsg($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-SuccessMsg($msg) { Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Write-ErrorMsg($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Start message
Write-InfoMsg '==== UVX Playwright System Preparation Started ====' 

# Check Python environment first
if (-not (Test-PythonEnvironment)) {
    Read-Host 'Press Enter to exit'
    exit 1
}

# Install Visual C++ Redistributable
Write-InfoMsg 'Installing system dependencies...'

# Check if Visual C++ Redistributable is already installed
$vcredist = Get-WmiObject -Class Win32_Product | Where-Object { $_.Name -like '*Visual C++*Redistributable*' -and ($_.Name -like '*2019*' -or $_.Name -like '*2022*') }

if (-not $vcredist) {
    try {
        # Download and install Visual C++ Redistributable 2022 x64
        $vcUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
        $vcPath = "$env:TEMP\vc_redist.x64.exe"
        
        Write-InfoMsg 'Downloading Visual C++ Redistributable 2022...'
        Invoke-WebRequest -Uri $vcUrl -OutFile $vcPath -UseBasicParsing
        
        Write-InfoMsg 'Installing Visual C++ Redistributable...'
        Start-Process -FilePath $vcPath -ArgumentList '/quiet', '/norestart' -Wait
        
        Remove-Item $vcPath -Force -ErrorAction SilentlyContinue
    } catch {
        Write-ErrorMsg 'Installation failed. Manual download: https://aka.ms/vs/17/release/vc_redist.x64.exe'
    }
} else {
    Write-InfoMsg 'Visual C++ Redistributable already installed'
}

# Check for Playwright cache (Windows uses AppData\Local)
$cachePattern = Join-Path $env:USERPROFILE 'AppData\Local\ms-playwright\chromium-*'
$cacheDirs = Get-ChildItem $cachePattern -Directory -ErrorAction SilentlyContinue
$validCache = $false
$foundCacheDir = $null

foreach ($cacheDir in $cacheDirs) {
    $chromePath = Join-Path $cacheDir 'chrome-win\chrome.exe'
    if (Test-Path $chromePath) {
        try {
            $versionOutput = & $chromePath --version 2>$null
            if ($versionOutput -match '(\d+\.\d+\.\d+\.\d+)') {
                $version = $matches[1]
                $versionParts = $version.Split('.') | ForEach-Object { [int]$_ }
                $minVersionParts = @(120, 0, 0, 0)
                
                $isNewer = $false
                for ($i = 0; $i -lt 4; $i++) {
                    if ($versionParts[$i] -gt $minVersionParts[$i]) {
                        $isNewer = $true
                        break
                    } elseif ($versionParts[$i] -lt $minVersionParts[$i]) {
                        break
                    }
                }
                
                if ($isNewer -or ($version -eq '120.0.0.0')) {
                    $validCache = $true
                    $foundCacheDir = $cacheDir
                    Write-SuccessMsg "Valid Playwright cache found: $version"
                    Write-Host "UVX will automatically use existing cache." -ForegroundColor Green
                    
                    # Configure environment for UVX to find the cache
                    $playwrightBrowsersPath = Join-Path $env:USERPROFILE 'AppData\Local\ms-playwright'
                    Write-InfoMsg "Configuring UVX environment for Playwright cache..."
                    Write-Host "  PLAYWRIGHT_BROWSERS_PATH=$playwrightBrowsersPath" -ForegroundColor Gray
                    
                    # Set environment variable for current session
                    [Environment]::SetEnvironmentVariable("PLAYWRIGHT_BROWSERS_PATH", $playwrightBrowsersPath, "User")
                    
                    Write-Host ""
                    Write-Host "Next step - Run UVX with cache environment:"
                    Write-Host "Method 1 (One-liner):" -ForegroundColor Green
                    Write-Host "  `$env:PLAYWRIGHT_BROWSERS_PATH='$playwrightBrowsersPath'; uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
                    Write-Host ""
                    Write-Host "Method 2 (Restart PowerShell after this script):" -ForegroundColor Green
                    Write-Host "  uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
                    Write-Host ""
                    Write-Host "Method 3 (Using cmd.exe):" -ForegroundColor Yellow
                    Write-Host "  set PLAYWRIGHT_BROWSERS_PATH=$playwrightBrowsersPath && uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor Gray
                    break
                } else {
                    Write-Host "Outdated Playwright cache: $version (required: 120.0.0.0+)" -ForegroundColor Yellow
                }
            }
        } catch {
            # Skip if version check fails
        }
    }
}

# Provide guidance based on cache status
if (-not $validCache) {
    Write-Host ""
    Write-Host "No Playwright cache found." -ForegroundColor Yellow
    $response = Read-Host "Install Chromium cache automatically? (y/N)"
    
    if ($response -match '^[Yy]') {
        $tempVenvDir = Join-Path $env:TEMP "playwright-install-$(Get-Random)"
        
        try {
            Write-InfoMsg "Installing Chromium cache automatically..."
            
            # Double-check Python environment before proceeding
            if (-not (Test-PythonEnvironment)) {
                throw "Python environment not ready"
            }
            
            # Create temporary virtual environment
            Write-InfoMsg "Creating temporary virtual environment..."
            
            & python -m venv $tempVenvDir
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create virtual environment. Please ensure Python 3.7+ with venv module is installed."
            }
            
            $pythonExe = Join-Path $tempVenvDir "Scripts\python.exe"
            $pipExe = Join-Path $tempVenvDir "Scripts\pip.exe"
            
            # Get Playwright version from requirements.txt
            $playwrightSpec = Get-PlaywrightVersionFromRequirements @(
                '..\requirements.txt',
                '.\requirements.txt',
                '..\dxt-packages\crawl4ai-dxt-correct\requirements.txt'
            )
            
            Write-InfoMsg "Installing Playwright... ($playwrightSpec)"
            
            & $pipExe install --quiet $playwrightSpec
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install Playwright"
            }
            
            Write-InfoMsg "Downloading Chromium..."
            
            & $pythonExe -m playwright install chromium
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install Chromium"
            }
            
            Write-SuccessMsg "Chromium cache installation completed!"
            
            # Configure environment for UVX to find the newly installed cache
            $playwrightBrowsersPath = Join-Path $env:USERPROFILE 'AppData\Local\ms-playwright'
            Write-InfoMsg "Configuring UVX environment for Playwright cache..."
            Write-Host "  PLAYWRIGHT_BROWSERS_PATH=$playwrightBrowsersPath" -ForegroundColor Gray
            
            # Set environment variable for current session
            [Environment]::SetEnvironmentVariable("PLAYWRIGHT_BROWSERS_PATH", $playwrightBrowsersPath, "User")
            
            Write-Host ""
            Write-Host "Next step - Run UVX with cache environment:"
            Write-Host "Method 1 (One-liner):" -ForegroundColor Green
            Write-Host "  `$env:PLAYWRIGHT_BROWSERS_PATH='$playwrightBrowsersPath'; uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
            Write-Host ""
            Write-Host "Method 2 (Restart PowerShell after this script):" -ForegroundColor Green
            Write-Host "  uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
            Write-Host ""
            Write-Host "Method 3 (Using cmd.exe):" -ForegroundColor Yellow
            Write-Host "  set PLAYWRIGHT_BROWSERS_PATH=$playwrightBrowsersPath && uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor Gray
            
        } catch {
            Write-ErrorMsg "Automatic installation failed: $($_.Exception.Message)"
            
            # Show manual instructions
            $playwrightSpec = Get-PlaywrightVersionFromRequirements @(
                '..\requirements.txt',
                '.\requirements.txt',
                '..\dxt-packages\crawl4ai-dxt-correct\requirements.txt'
            )
            
            Write-Host ""
            Write-Host "Manual Chromium cache installation:" -ForegroundColor Yellow
            Write-Host "  python -m venv venv" -ForegroundColor White
            Write-Host "  venv\Scripts\activate" -ForegroundColor White
            Write-Host "  pip install $playwrightSpec" -ForegroundColor White
            Write-Host "  python -m playwright install chromium" -ForegroundColor White
            Write-Host ""
            Write-Host "UVX execution with cache environment:" -ForegroundColor Yellow
            $playwrightBrowsersPath = Join-Path $env:USERPROFILE 'AppData\Local\ms-playwright'
            Write-Host "Method 1 (One-liner):" -ForegroundColor Green
            Write-Host "  `$env:PLAYWRIGHT_BROWSERS_PATH='$playwrightBrowsersPath'; uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
            Write-Host ""
            Write-Host "Method 2 (Using cmd.exe):" -ForegroundColor Yellow
            Write-Host "  set PLAYWRIGHT_BROWSERS_PATH=$playwrightBrowsersPath && uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
        } finally {
            # Cleanup
            if (Test-Path $tempVenvDir) {
                Remove-Item $tempVenvDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    } else {
        # Show manual instructions
        $playwrightSpec = Get-PlaywrightVersionFromRequirements @(
            '..\requirements.txt',
            '.\requirements.txt',
            '..\dxt-packages\crawl4ai-dxt-correct\requirements.txt'
        )
        
        Write-Host ""
        Write-Host "Manual Chromium cache installation:" -ForegroundColor Yellow
        Write-Host "  python -m venv venv" -ForegroundColor White
        Write-Host "  venv\Scripts\activate" -ForegroundColor White
        Write-Host "  pip install $playwrightSpec" -ForegroundColor White
        Write-Host "  python -m playwright install chromium" -ForegroundColor White
        Write-Host ""
        Write-Host "UVX execution with cache environment:" -ForegroundColor Yellow
        $playwrightBrowsersPath = Join-Path $env:USERPROFILE 'AppData\Local\ms-playwright'
        Write-Host "Method 1 (One-liner):" -ForegroundColor Green
        Write-Host "  `$env:PLAYWRIGHT_BROWSERS_PATH='$playwrightBrowsersPath'; uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
        Write-Host ""
        Write-Host "Method 2 (Using cmd.exe):" -ForegroundColor Yellow
        Write-Host "  set PLAYWRIGHT_BROWSERS_PATH=$playwrightBrowsersPath && uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp" -ForegroundColor White
    }
}

# Success message
Write-SuccessMsg 'UVX Playwright system preparation complete!'

# Troubleshooting information
Write-Host ""
Write-Host "=== Troubleshooting Information ===" -ForegroundColor Cyan
Write-Host "If UVX cannot find Playwright cache, run these diagnostic commands:" -ForegroundColor Yellow

Write-Host ""
Write-Host "# Check cache directory:" -ForegroundColor Gray
Write-Host "`$cacheDir = `"$env:USERPROFILE\AppData\Local\ms-playwright`"" -ForegroundColor White
Write-Host "Test-Path `$cacheDir" -ForegroundColor White

Write-Host ""
Write-Host "# List Chromium directories:" -ForegroundColor Gray
Write-Host "Get-ChildItem `"$env:USERPROFILE\AppData\Local\ms-playwright\chromium-*`" -Directory" -ForegroundColor White

Write-Host ""
Write-Host "# Verify Chrome executable:" -ForegroundColor Gray
Write-Host "Get-ChildItem `"$env:USERPROFILE\AppData\Local\ms-playwright\chromium-*\chrome-win\chrome.exe`"" -ForegroundColor White

Write-Host ""
Write-Host "# Current environment variable:" -ForegroundColor Gray
Write-Host "Write-Host `"PLAYWRIGHT_BROWSERS_PATH: `$env:PLAYWRIGHT_BROWSERS_PATH`"" -ForegroundColor White

Write-Host ""
Read-Host 'Press Enter to exit'