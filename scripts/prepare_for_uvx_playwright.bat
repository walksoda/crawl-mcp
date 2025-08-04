@echo off
REM ==========================================
REM UVX Playwright System Preparation Script (Windows)
REM Purpose: Prepare system dependencies for UVX crawl4ai-dxt-correct execution
REM Language: Auto-detect (English default, Japanese if LANG=ja* or CRAWL4AI_LANG=ja)
REM Usage: Run as Administrator - scripts/prepare_for_uvx_playwright.bat
REM ==========================================

REM PowerShell script body
powershell -NoProfile -ExecutionPolicy Bypass -Command "& {
$ErrorActionPreference = 'Stop'

# Language detection
$lang = if ($env:LANG -match '^ja' -or $env:CRAWL4AI_LANG -eq 'ja') { 'ja' } else { 'en' }

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
        if ($lang -eq 'ja') {
            Write-ErrorMsg 'Python環境に問題があります:'
            Write-Host "  - Python 3.7+が必要です (現在: $(if($pythonVersion) { $pythonVersion } else { '未インストール' }))"
            Write-Host '  - 以下からPythonをインストールしてください:'
            Write-Host '    https://www.python.org/downloads/windows/'
            Write-Host '  - インストール時に "Add Python to PATH" をチェックしてください'
        } else {
            Write-ErrorMsg 'Python environment issues detected:'
            Write-Host "  - Python 3.7+ required (current: $(if($pythonVersion) { $pythonVersion } else { 'not installed' }))"
            Write-Host '  - Please install Python from:'
            Write-Host '    https://www.python.org/downloads/windows/'
            Write-Host '  - Make sure to check "Add Python to PATH" during installation'
        }
        return $false
    }
}

# Localized message functions
function Get-LocalizedMsg($key) {
    $messages = @{
        'admin_required' = @{
            'ja' = '管理者権限で再実行してください。'
            'en' = 'Please run with administrator privileges.'
        }
        'starting' = @{
            'ja' = '==== UVX Playwright システム準備開始 ===='
            'en' = '==== UVX Playwright System Preparation Started ===='
        }
        'installing_deps' = @{
            'ja' = 'システム依存関係をインストール中...'
            'en' = 'Installing system dependencies...'
        }
        'success' = @{
            'ja' = 'UVX Playwright用システム準備完了！'
            'en' = 'UVX Playwright system preparation complete!'
        }
        'next_steps' = @{
            'ja' = @'

次のステップ:
1. 手動でChromiumキャッシュをインストール（必要に応じて）
2. UVX実行: uvx --from crawl4ai-dxt-correct crawl4ai_mcp

Chromiumキャッシュ手動インストール手順:
  python -m venv venv
  venv\Scripts\activate
  pip install playwright
  python -m playwright install chromium
'@
            'en' = @'

Next steps:
1. Manually install Chromium cache (if needed)
2. Run UVX: uvx --from crawl4ai-dxt-correct crawl4ai_mcp

Manual Chromium cache installation steps:
  python -m venv venv
  venv\Scripts\activate
  pip install playwright
  python -m playwright install chromium
'@
        }
        'install_failed' = @{
            'ja' = 'Visual C++ Redistributableのインストール手順をご確認ください。'
            'en' = 'Please check Visual C++ Redistributable installation instructions.'
        }
    }
    
    return $messages[$key][$lang]
}

function Write-Info($msg) { Write-Host \"[INFO] $msg\" -ForegroundColor Cyan }
function Write-Success($msg) { Write-Host \"[SUCCESS] $msg\" -ForegroundColor Green }
function Write-ErrorMsg($msg) { Write-Host \"[ERROR] $msg\" -ForegroundColor Red }

# Administrator check
If (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-ErrorMsg (Get-LocalizedMsg 'admin_required')
    Pause
    Exit 1
}

Write-Info (Get-LocalizedMsg 'starting')

# Check Python environment first
if (-not (Test-PythonEnvironment)) {
    Pause
    Exit 1
}

# Install essential Visual C++ Redistributable for Playwright browsers
Write-Info (Get-LocalizedMsg 'installing_deps')

# Check if Visual C++ Redistributable is already installed
$vcredist = Get-WmiObject -Class Win32_Product | Where-Object { $_.Name -like '*Visual C++*Redistributable*' -and $_.Name -like '*2019*' -or $_.Name -like '*2022*' }

if (-not $vcredist) {
    try {
        # Download and install Visual C++ Redistributable 2022 x64
        $vcUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
        $vcPath = \"$env:TEMP\vc_redist.x64.exe\"
        
        Write-Info 'Downloading Visual C++ Redistributable 2022...'
        Invoke-WebRequest -Uri $vcUrl -OutFile $vcPath -UseBasicParsing
        
        Write-Info 'Installing Visual C++ Redistributable...'
        Start-Process -FilePath $vcPath -ArgumentList '/quiet', '/norestart' -Wait
        
        Remove-Item $vcPath -Force -ErrorAction SilentlyContinue
    } catch {
        Write-ErrorMsg (Get-LocalizedMsg 'install_failed')
        Write-Host 'Manual download: https://aka.ms/vs/17/release/vc_redist.x64.exe' -ForegroundColor Yellow
    }
} else {
    Write-Info 'Visual C++ Redistributable already installed'
}

# Check for Playwright cache and offer automatic installation
function Test-PlaywrightCache {
    $cachePattern = Join-Path $env:USERPROFILE '.cache\ms-playwright\chromium-*'
    $cacheDirs = Get-ChildItem $cachePattern -Directory -ErrorAction SilentlyContinue
    
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
                        if ($lang -eq 'ja') {
                            Write-Success "有効なPlaywrightキャッシュを検出: $version"
                            Write-Host "UVXは既存のキャッシュを自動的に使用します。"
                            Write-Host "`n🎯 UVX実行方法:"
                            Write-Host "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
                        } else {
                            Write-Success "Valid Playwright cache found: $version"
                            Write-Host "UVX will automatically use existing cache."
                            Write-Host "`n🎯 UVX execution:"
                            Write-Host "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
                        }
                        return $true
                    } else {
                        if ($lang -eq 'ja') {
                            Write-Host "古いPlaywrightキャッシュ: $version < 120.0.0.0" -ForegroundColor Yellow
                        } else {
                            Write-Host "Outdated Playwright cache: $version < 120.0.0.0" -ForegroundColor Yellow
                        }
                    }
                }
            } catch {
                # Skip if version check fails
            }
        }
    }
    
    # No valid cache found, offer installation
    if ($lang -eq 'ja') {
        Write-Host "`nPlaywrightキャッシュがありません。"
        $response = Read-Host "Chromiumキャッシュを自動インストールしますか？ (y/N)"
    } else {
        Write-Host "`nNo Playwright cache found."
        $response = Read-Host "Install Chromium cache automatically? (y/N)"
    }
    
    if ($response -match '^[Yy]') {
        Install-ChromiumCache
    } else {
        Show-ManualInstructions
    }
    
    return $false
}

function Install-ChromiumCache {
    $tempVenvDir = Join-Path $env:TEMP "playwright-install-$(Get-Random)"
    
    try {
        if ($lang -eq 'ja') {
            Write-Info "Chromiumキャッシュを自動インストール中..."
        } else {
            Write-Info "Installing Chromium cache automatically..."
        }
        
        # Double-check Python environment before proceeding
        if (-not (Test-PythonEnvironment)) {
            throw "Python environment not ready"
        }
        
        # Create temporary virtual environment
        if ($lang -eq 'ja') {
            Write-Info "一時仮想環境を作成中..."
        } else {
            Write-Info "Creating temporary virtual environment..."
        }
        
        & python -m venv $tempVenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment. Please ensure Python 3.7+ with venv module is installed."
        }
        
        $pythonExe = Join-Path $tempVenvDir "Scripts\python.exe"
        $pipExe = Join-Path $tempVenvDir "Scripts\pip.exe"
        
        if ($lang -eq 'ja') {
            Write-Info "Playwrightをインストール中..."
        } else {
            Write-Info "Installing Playwright..."
        }
        
        & $pipExe install --quiet playwright
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Playwright"
        }
        
        if ($lang -eq 'ja') {
            Write-Info "Chromiumをダウンロード中..."
        } else {
            Write-Info "Downloading Chromium..."
        }
        
        & $pythonExe -m playwright install chromium
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Chromium"
        }
        
        if ($lang -eq 'ja') {
            Write-Success "Chromiumキャッシュのインストールが完了しました！"
            Write-Host "`n🎯 UVX実行方法:"
            Write-Host "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
        } else {
            Write-Success "Chromium cache installation completed!"
            Write-Host "`n🎯 UVX execution:"
            Write-Host "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
        }
        
    } catch {
        if ($lang -eq 'ja') {
            Write-ErrorMsg "自動インストールに失敗しました: $($_.Exception.Message)"
        } else {
            Write-ErrorMsg "Automatic installation failed: $($_.Exception.Message)"
        }
        Show-ManualInstructions
    } finally {
        # Cleanup
        if (Test-Path $tempVenvDir) {
            Remove-Item $tempVenvDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Show-ManualInstructions {
    if ($lang -eq 'ja') {
        Write-Host "`n📋 手動でChromiumキャッシュをインストール:"
        Write-Host "  python -m venv venv"
        Write-Host "  venv\Scripts\activate"
        Write-Host "  pip install playwright"
        Write-Host "  python -m playwright install chromium"
        Write-Host "`n🎯 インストール後のUVX実行:"
        Write-Host "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
    } else {
        Write-Host "`n📋 Manual Chromium cache installation:"
        Write-Host "  python -m venv venv"
        Write-Host "  venv\Scripts\activate"
        Write-Host "  pip install playwright"
        Write-Host "  python -m playwright install chromium"
        Write-Host "`n🎯 UVX execution after installation:"
        Write-Host "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
    }
}

Write-Success (Get-LocalizedMsg 'success')

# Check Playwright cache and offer installation
Test-PlaywrightCache

Pause
}"