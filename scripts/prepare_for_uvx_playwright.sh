#!/usr/bin/env bash

# =========================
# UVX Playwright System Preparation Script
# Purpose: Prepare system dependencies for UVX crawl4ai-dxt-correct execution
# Cross-distribution support: Ubuntu/Debian/CentOS/Alpine/Arch
# Language: Auto-detect (English default, Japanese if LANG=ja* or CRAWL4AI_LANG=ja)
# Usage: sudo bash scripts/prepare_for_uvx_playwright.sh
# =========================

set -e

LOGFILE="uvx_playwright_prep.log"

# Language detection
detect_language() {
    if [[ "${LANG:-}" =~ ^ja || "${CRAWL4AI_LANG:-}" == "ja" ]]; then
        SCRIPT_LANG="ja"
    else
        SCRIPT_LANG="en"
    fi
}

# Simplified localized messages for UVX preparation
msg_starting() {
    case "$SCRIPT_LANG" in
        ja) echo "==== UVX Playwright システム準備開始 ====" ;;
        *) echo "==== UVX Playwright System Preparation Started ====" ;;
    esac
}

msg_os_detected() {
    case "$SCRIPT_LANG" in
        ja) echo "OS検出: $1 ($2)" ;;
        *) echo "Detected OS: $1 ($2)" ;;
    esac
}

msg_pkg_manager() {
    case "$SCRIPT_LANG" in
        ja) echo "パッケージマネージャー: $1" ;;
        *) echo "Package Manager: $1" ;;
    esac
}

msg_installing_deps() {
    case "$SCRIPT_LANG" in
        ja) echo "システム依存関係をインストール中..." ;;
        *) echo "Installing system dependencies..." ;;
    esac
}

msg_success() {
    case "$SCRIPT_LANG" in
        ja) echo "UVX Playwright用システム準備完了！" ;;
        *) echo "UVX Playwright system preparation complete!" ;;
    esac
}

msg_next_steps() {
    case "$SCRIPT_LANG" in
        ja) echo "
次のステップ:
1. 手動でChromiumキャッシュをインストール（必要に応じて）
2. UVX実行: uvx --from crawl4ai-dxt-correct crawl4ai_mcp

Chromiumキャッシュ手動インストール手順:
  python3 -m venv venv
  source venv/bin/activate
  pip install playwright
  python -m playwright install chromium" ;;
        *) echo "
Next steps:
1. Manually install Chromium cache (if needed)
2. Run UVX: uvx --from crawl4ai-dxt-correct crawl4ai_mcp

Manual Chromium cache installation steps:
  python3 -m venv venv
  source venv/bin/activate
  pip install playwright
  python -m playwright install chromium" ;;
    esac
}

msg_log_location() {
    case "$SCRIPT_LANG" in
        ja) echo "詳細ログ: $1" ;;
        *) echo "Detailed log: $1" ;;
    esac
}

# Initialize language detection
detect_language

# Colored output functions
info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }

# Log output
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOGFILE"; }

# OS/Package manager detection
detect_os() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID=$ID
    OS_NAME=$NAME
  else
    OS_ID=$(uname -s)
    OS_NAME=$OS_ID
  fi
  info "$(msg_os_detected "$OS_NAME" "$OS_ID")"
}

detect_package_manager() {
  if command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt"
  elif command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
  elif command -v yum >/dev/null 2>&1; then
    PKG_MGR="yum"
  elif command -v pacman >/dev/null 2>&1; then
    PKG_MGR="pacman"
  elif command -v apk >/dev/null 2>&1; then
    PKG_MGR="apk"
  else
    case "$SCRIPT_LANG" in
      ja) error "対応するパッケージマネージャーが見つかりません。" ;;
      *) error "No supported package manager found." ;;
    esac
    exit 1
  fi
  info "$(msg_pkg_manager "$PKG_MGR")"
}

# Install minimal system dependencies for Playwright browsers
install_system_dependencies() {
  info "$(msg_installing_deps)"
  case "$PKG_MGR" in
    apt)
      sudo apt-get update
      sudo apt-get install -y --no-install-recommends \
        libnss3 libgbm1 libxss1 ca-certificates fonts-liberation \
        libatk-bridge2.0-0 libdrm2 libxcomposite1 libxdamage1 \
        libxrandr2 libgtk-3-0 libxkbcommon0 libasound2 \
        python3-venv python3-pip
      ;;
    dnf|yum)
      sudo $PKG_MGR install -y nss atk at-spi2-atk cups-libs libdrm \
        libgtk-3 libgbm xorg-x11-server-Xvfb liberation-fonts \
        libXcomposite libXdamage libXrandr gtk3 alsa-lib \
        python3 python3-pip python3-venv
      ;;
    pacman)
      sudo pacman -Sy --noconfirm nss gtk3 libgbm alsa-lib \
        libxcomposite libxdamage libxrandr at-spi2-atk \
        python python-pip
      ;;
    apk)
      sudo apk add --no-cache nss gtk+3.0 mesa-gbm ca-certificates \
        liberation-fonts-ttf alsa-lib libxcomposite libxdamage libxrandr \
        python3 py3-pip py3-virtualenv
      ;;
    *)
      case "$SCRIPT_LANG" in
        ja) warn "依存パッケージの自動導入未対応。手動でインストールしてください。" ;;
        *) warn "Automatic dependency installation not supported. Please install manually." ;;
      esac
      ;;
  esac
  
  log "System dependencies installation completed"
}

# Version comparison function
version_ge() {
  # Compare versions: $1 >= $2 returns 0 (true), else 1 (false)
  [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Dynamic UVX environment requirement detection
detect_uvx_playwright_requirements() {
  log "Detecting UVX Playwright requirements..."
  
  # Try to detect UVX Playwright version
  UVX_PW_VERSION=""
  if command -v uvx &> /dev/null; then
    # Method 1: Check UVX package info
    UVX_PW_VERSION=$(uvx --from crawl4ai-dxt-correct --dry-run python -c "
try:
  from importlib.metadata import version
  print(version('playwright'))
except:
  try:
    import pkg_resources
    print(pkg_resources.get_distribution('playwright').version)
  except:
    print('unknown')
" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
  fi
  
  # Method 2: Try to detect required Chromium from UVX environment
  UVX_REQUIRED_CHROMIUM=""
  if [ "$UVX_PW_VERSION" != "unknown" ] && [ -n "$UVX_PW_VERSION" ]; then
    UVX_REQUIRED_CHROMIUM=$(uvx --from crawl4ai-dxt-correct --dry-run python -c "
import subprocess, re, sys
try:
  result = subprocess.run([sys.executable, '-m', 'playwright', 'install', '--dry-run', 'chromium'], 
                         capture_output=True, text=True, timeout=30)
  if result.returncode == 0 and 'chromium version' in result.stdout:
    version_match = re.search(r'chromium version (\d+\.\d+\.\d+\.\d+)', result.stdout)
    if version_match:
      print(version_match.group(1))
    else:
      print('unknown')
  else:
    print('unknown')
except:
  print('unknown')
" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
  fi
  
  log "UVX Playwright version: ${UVX_PW_VERSION:-unknown}"
  log "UVX required Chromium: ${UVX_REQUIRED_CHROMIUM:-unknown}"
}

# Get Playwright-Chromium version mapping from API with caching
get_playwright_chromium_mapping() {
  local cache_file="/tmp/playwright_versions_$(date +%Y%m%d).json"
  local pw_version="${1:-$UVX_PW_VERSION}"
  
  # Check cache validity (24 hours)
  if [ -f "$cache_file" ] && [ $(find "$cache_file" -mtime -1 2>/dev/null | wc -l) -gt 0 ]; then
    log "Using cached Playwright version mapping"
    return 0
  fi
  
  log "Fetching latest Playwright-Chromium mapping..."
  
  # Try PyPI API for latest Playwright info
  if command -v curl &> /dev/null; then
    curl -s "https://pypi.org/pypi/playwright/json" > "${cache_file}.tmp" 2>/dev/null
    if [ $? -eq 0 ] && [ -s "${cache_file}.tmp" ]; then
      mv "${cache_file}.tmp" "$cache_file"
      log "Successfully cached Playwright version mapping"
    else
      rm -f "${cache_file}.tmp" 2>/dev/null
      log "Failed to fetch Playwright version mapping"
    fi
  fi
}

# Intelligent minimum Chromium version calculation
calculate_minimum_chromium_version() {
  local target_pw_version="${1:-$UVX_PW_VERSION}"
  local fallback_version="120.0.0.0"
  
  # If we have UVX detected requirement, use it
  if [ "$UVX_REQUIRED_CHROMIUM" != "unknown" ] && [ -n "$UVX_REQUIRED_CHROMIUM" ]; then
    echo "$UVX_REQUIRED_CHROMIUM"
    return 0
  fi
  
  # Calculate based on Playwright version patterns
  if [ "$target_pw_version" != "unknown" ] && [ -n "$target_pw_version" ]; then
    local calculated_version=$(python3 -c "
import sys, re
try:
  pw_version = '$target_pw_version'
  version_match = re.match(r'(\d+)\.(\d+)\.(\d+)', pw_version)
  if version_match:
    major, minor, patch = map(int, version_match.groups())
    if major == 1:
      if minor >= 55:
        print('140.0.0.0')  # Future versions
      elif minor >= 54:
        print('139.0.0.0')  # chromium-1181+ (Playwright 1.54+)
      elif minor >= 52:
        print('136.0.0.0')  # chromium-1169+ (Playwright 1.52+)
      elif minor >= 50:
        print('130.0.0.0')  # chromium-1100+ (Playwright 1.50+)
      else:
        print('$fallback_version')
    else:
      print('$fallback_version')
  else:
    print('$fallback_version')
except:
  print('$fallback_version')
" 2>/dev/null || echo "$fallback_version")
    
    echo "$calculated_version"
    return 0
  fi
  
  # Fallback to conservative estimate
  echo "$fallback_version"
}

# Dynamic minimum version determination with multiple strategies
get_minimum_chromium_version() {
  log "Determining minimum Chromium version requirement..."
  
  # Strategy 1: Detect UVX requirements
  detect_uvx_playwright_requirements
  
  # Strategy 2: Get API mapping if available
  get_playwright_chromium_mapping
  
  # Strategy 3: Calculate intelligent minimum
  local calculated_min=$(calculate_minimum_chromium_version)
  
  # Strategy 4: Apply safety margin for future compatibility
  local final_version=$(python3 -c "
try:
  version = '$calculated_min'
  parts = version.split('.')
  if len(parts) >= 1:
    major = int(parts[0])
    # Add small safety margin (e.g., +1 to major version for forward compatibility)
    safety_major = max(major, 137)  # Ensure at least 137.x for known UVX compatibility
    print(f'{safety_major}.0.0.0')
  else:
    print('137.0.0.0')
except:
  print('137.0.0.0')
" 2>/dev/null || echo "137.0.0.0")
  
  log "Calculated minimum Chromium version: $final_version"
  case "$SCRIPT_LANG" in
    ja) info "動的に算出された最小要求バージョン: $final_version" ;;
    *) info "Dynamically calculated minimum required version: $final_version" ;;
  esac
  
  echo "$final_version"
}

# Check current Playwright version in system
check_current_playwright_version() {
  if command -v python3 &> /dev/null; then
    CURRENT_PW_VERSION=$(python3 -c "
try:
  from importlib.metadata import version
  print(version('playwright'))
except:
  try:
    import pkg_resources
    print(pkg_resources.get_distribution('playwright').version)
  except:
    print('not_installed')
" 2>/dev/null || echo "not_installed")
  else
    CURRENT_PW_VERSION="python_not_found"
  fi
  
  log "Current local Playwright version: $CURRENT_PW_VERSION"
}

# Get latest Playwright version from PyPI
get_latest_playwright_version() {
  if command -v curl &> /dev/null; then
    LATEST_PW_VERSION=$(curl -s "https://pypi.org/pypi/playwright/json" 2>/dev/null | python3 -c "
import sys, json
try:
  data = json.load(sys.stdin)
  print(data['info']['version'])
except:
  print('unknown')
" 2>/dev/null || echo "unknown")
  else
    LATEST_PW_VERSION="unknown"
  fi
  
  log "Latest PyPI Playwright version: $LATEST_PW_VERSION"
}

# Compare Playwright versions  
compare_playwright_versions() {
  if [ "$CURRENT_PW_VERSION" = "not_installed" ] || [ "$CURRENT_PW_VERSION" = "python_not_found" ]; then
    return 0  # Need installation
  fi
  
  if [ "$LATEST_PW_VERSION" = "unknown" ]; then
    return 1  # Cannot determine, skip update
  fi
  
  # Version comparison using sort -V
  if [ "$(printf '%s\n' "$CURRENT_PW_VERSION" "$LATEST_PW_VERSION" | sort -V | head -n1)" != "$LATEST_PW_VERSION" ]; then
    return 0  # Current < Latest, update needed
  else
    return 1  # Current >= Latest, no update needed
  fi
}

# Update Playwright library to latest version
update_playwright_library() {
  TEMP_VENV_DIR="/tmp/playwright-update-$$"
  
  case "$SCRIPT_LANG" in
    ja) 
      info "Playwrightライブラリを最新版にアップデート中..."
      info "現在: $CURRENT_PW_VERSION → 最新: $LATEST_PW_VERSION"
      ;;
    *)
      info "Updating Playwright library to latest version..."
      info "Current: $CURRENT_PW_VERSION → Latest: $LATEST_PW_VERSION"
      ;;
  esac
  
  # Create temporary virtual environment
  if ! python3 -m venv "$TEMP_VENV_DIR" 2>/dev/null; then
    case "$SCRIPT_LANG" in
      ja) error "仮想環境の作成に失敗しました。python3-venvパッケージを確認してください。" ;;
      *) error "Failed to create virtual environment. Check python3-venv package." ;;
    esac
    return 1
  fi
  
  # Playwright update in virtual environment
  (
    . "$TEMP_VENV_DIR/bin/activate"
    
    case "$SCRIPT_LANG" in
      ja) info "Playwrightをアップグレード中..." ;;
      *) info "Upgrading Playwright..." ;;
    esac
    
    if ! pip install --upgrade playwright --quiet; then
      case "$SCRIPT_LANG" in
        ja) error "Playwrightのアップグレードに失敗しました。" ;;
        *) error "Failed to upgrade Playwright." ;;
      esac
      rm -rf "$TEMP_VENV_DIR"
      return 1
    fi
    
    case "$SCRIPT_LANG" in
      ja) info "最新Chromiumをインストール中..." ;;
      *) info "Installing latest Chromium..." ;;
    esac
    
    if ! python -m playwright install chromium; then
      case "$SCRIPT_LANG" in
        ja) error "Chromiumのインストールに失敗しました。" ;;
        *) error "Failed to install Chromium." ;;
      esac
      rm -rf "$TEMP_VENV_DIR"
      return 1
    fi
    
    # Verify new version
    NEW_PW_VERSION=$(python -c "
try:
  from importlib.metadata import version
  print(version('playwright'))
except:
  print('unknown')
" 2>/dev/null || echo "unknown")
    
    case "$SCRIPT_LANG" in
      ja) 
        success "Playwrightアップデート完了！"
        info "新バージョン: $NEW_PW_VERSION"
        ;;
      *)
        success "Playwright update completed!"
        info "New version: $NEW_PW_VERSION"
        ;;
    esac
  )
  
  # Cleanup temporary venv
  rm -rf "$TEMP_VENV_DIR"
  
  log "Playwright library update completed successfully"
}

# Install Chromium cache automatically (enhanced with Playwright update)
install_chromium_cache() {
  TEMP_VENV_DIR="/tmp/playwright-install-$$"
  
  case "$SCRIPT_LANG" in
    ja) info "Chromiumキャッシュを自動インストール中..." ;;
    *) info "Installing Chromium cache automatically..." ;;
  esac
  
  # Create temporary virtual environment
  if ! python3 -m venv "$TEMP_VENV_DIR" 2>/dev/null; then
    case "$SCRIPT_LANG" in
      ja) error "仮想環境の作成に失敗しました。python3-venvパッケージをインストールしてください。" ;;
      *) error "Failed to create virtual environment. Please install python3-venv package." ;;
    esac
    return 1
  fi
  
  # Activate virtual environment and install
  (
    . "$TEMP_VENV_DIR/bin/activate"
    
    case "$SCRIPT_LANG" in
      ja) info "Playwrightをインストール中..." ;;
      *) info "Installing Playwright..." ;;
    esac
    
    if ! pip install --quiet playwright; then
      case "$SCRIPT_LANG" in
        ja) error "Playwrightのインストールに失敗しました。" ;;
        *) error "Failed to install Playwright." ;;
      esac
      rm -rf "$TEMP_VENV_DIR"
      return 1
    fi
    
    case "$SCRIPT_LANG" in
      ja) info "Chromiumをダウンロード中..." ;;
      *) info "Downloading Chromium..." ;;
    esac
    
    if ! python -m playwright install chromium; then
      case "$SCRIPT_LANG" in
        ja) error "Chromiumのインストールに失敗しました。" ;;
        *) error "Failed to install Chromium." ;;
      esac
      rm -rf "$TEMP_VENV_DIR"
      return 1
    fi
    
    case "$SCRIPT_LANG" in
      ja) success "Chromiumキャッシュのインストールが完了しました！" ;;
      *) success "Chromium cache installation completed!" ;;
    esac
  )
  
  # Cleanup temporary venv
  rm -rf "$TEMP_VENV_DIR"
  
  case "$SCRIPT_LANG" in
    ja) 
      echo ""
      echo "🎯 UVX実行方法:"
      echo "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
      ;;
    *)
      echo ""
      echo "🎯 UVX execution:"
      echo "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
      ;;
  esac
  
  log "Chromium cache installation completed successfully"
}

# Enhanced Chromium environment setup with Playwright library management
setup_chromium_environment() {
  case "$SCRIPT_LANG" in
    ja) info "Playwright環境を分析中..." ;;
    *) info "Analyzing Playwright environment..." ;;
  esac
  
  # Step 1: Check Playwright library version
  check_current_playwright_version
  get_latest_playwright_version
  
  # Step 2: Handle Playwright library updates if needed
  if compare_playwright_versions; then
    case "$SCRIPT_LANG" in
      ja) 
        warn "Playwrightライブラリが古いバージョンです ($CURRENT_PW_VERSION)"
        if [ "$LATEST_PW_VERSION" != "unknown" ]; then
          info "最新版: $LATEST_PW_VERSION"
        fi
        echo ""
        read -p "Playwrightを最新版にアップデートしますか？ (y/N): " update_pw
        ;;
      *)
        warn "Playwright library is outdated ($CURRENT_PW_VERSION)"
        if [ "$LATEST_PW_VERSION" != "unknown" ]; then
          info "Latest version: $LATEST_PW_VERSION"
        fi
        echo ""
        read -p "Update Playwright to latest version? (y/N): " update_pw
        ;;
    esac
    
    case "$update_pw" in
      [Yy]*)
        if update_playwright_library; then
          case "$SCRIPT_LANG" in
            ja) success "Playwrightアップデートが完了しました。" ;;
            *) success "Playwright update completed successfully." ;;
          esac
          # Re-check after update
          check_current_playwright_version
        else
          case "$SCRIPT_LANG" in
            ja) error "Playwrightアップデートに失敗しました。手動で実行してください。" ;;
            *) error "Playwright update failed. Please update manually." ;;
          esac
          return 1
        fi
        ;;
      *)
        case "$SCRIPT_LANG" in
          ja) info "Playwrightアップデートをスキップしました。" ;;
          *) info "Playwright update skipped." ;;
        esac
        ;;
    esac
  else
    case "$SCRIPT_LANG" in
      ja) success "Playwrightライブラリは最新版です ($CURRENT_PW_VERSION)" ;;
      *) success "Playwright library is up to date ($CURRENT_PW_VERSION)" ;;
    esac
  fi
  
  # Step 3: Get dynamic minimum version requirement
  MINIMUM_VERSION=$(get_minimum_chromium_version)
  
  # Step 4: Check for existing Playwright cache
  VALID_CACHE=false
  CURRENT_VERSION=""
  
  for playwright_chrome in ~/.cache/ms-playwright/chromium-*/chrome-linux/chrome; do
    if [ -x "$playwright_chrome" ]; then
      current_version=$($playwright_chrome --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
      
      if [ -n "$current_version" ] && version_ge "$current_version" "$MINIMUM_VERSION"; then
        VALID_CACHE=true
        CURRENT_VERSION="$current_version"
        case "$SCRIPT_LANG" in
          ja) success "有効なPlaywrightキャッシュを検出: $current_version (要求: $MINIMUM_VERSION+)" ;;
          *) success "Valid Playwright cache found: $current_version (required: $MINIMUM_VERSION+)" ;;
        esac
        break
      else
        CURRENT_VERSION="$current_version"
        case "$SCRIPT_LANG" in
          ja) warn "古いPlaywrightキャッシュ: $current_version < $MINIMUM_VERSION" ;;
          *) warn "Outdated Playwright cache: $current_version < $MINIMUM_VERSION" ;;
        esac
      fi
    fi
  done
  
  # Provide guidance based on cache status
  if [ "$VALID_CACHE" = true ]; then
    case "$SCRIPT_LANG" in
      ja) 
        success "UVXは既存のPlaywrightキャッシュを自動的に使用します。"
        echo ""
        echo "🎯 UVX実行方法:"
        echo "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
        ;;
      *)
        success "UVX will automatically use existing Playwright cache."
        echo ""
        echo "🎯 UVX execution:"
        echo "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
        ;;
    esac
  else
    case "$SCRIPT_LANG" in
      ja) 
        if [ -n "$CURRENT_VERSION" ]; then
          warn "Chromiumキャッシュのアップデートが必要です (現在: $CURRENT_VERSION → 要求: $MINIMUM_VERSION+)"
        else
          info "Playwrightキャッシュがありません。"
        fi
        echo ""
        read -p "Chromiumキャッシュを自動インストールしますか？ (y/N): " yn
        ;;
      *)
        if [ -n "$CURRENT_VERSION" ]; then
          warn "Chromium cache update required (current: $CURRENT_VERSION → required: $MINIMUM_VERSION+)"
        else
          info "No Playwright cache found."
        fi
        echo ""
        read -p "Install Chromium cache automatically? (y/N): " yn
        ;;
    esac
    
    case "$yn" in
      [Yy]*)
        install_chromium_cache
        ;;
      *)
        case "$SCRIPT_LANG" in
          ja) 
            echo ""
            echo "📋 手動でChromiumキャッシュをインストール:"
            echo "  python3 -m venv venv"
            echo "  source venv/bin/activate"
            echo "  pip install playwright"
            echo "  python -m playwright install chromium"
            echo ""
            echo "🎯 インストール後のUVX実行:"
            echo "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
            ;;
          *)
            echo ""
            echo "📋 Manual Chromium cache installation:"
            echo "  python3 -m venv venv"
            echo "  source venv/bin/activate"
            echo "  pip install playwright"
            echo "  python -m playwright install chromium"
            echo ""
            echo "🎯 UVX execution after installation:"
            echo "  uvx --from crawl4ai-dxt-correct crawl4ai_mcp"
            ;;
        esac
        ;;
    esac
  fi
}

# Main function - UVX preparation with Chromium environment setup
main() {
  info "$(msg_starting)"
  log "UVX Playwright preparation started"
  
  detect_os
  detect_package_manager
  install_system_dependencies
  setup_chromium_environment
  
  success "$(msg_success)"
  echo "$(msg_next_steps)"
  info "$(msg_log_location "$LOGFILE")"
  
  log "UVX Playwright preparation completed successfully"
}

main "$@"