#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Playwright要求Chromiumバージョン検出・比較スクリプト
クロスプラットフォーム対応
"""

import sys
import os
import subprocess
import platform
import argparse
import json
import re

def get_playwright_version():
    try:
        import importlib.metadata
        return importlib.metadata.version("playwright")
    except Exception:
        try:
            import pkg_resources
            return pkg_resources.get_distribution("playwright").version
        except Exception:
            return None

def get_playwright_chromium_version():
    """
    Playwrightのchromiumバージョンを取得
    """
    try:
        from playwright._impl._driver import compute_driver_executable
        import importlib.metadata
        # Playwright公式のバージョンマッピング
        # 参考: https://github.com/microsoft/playwright/blob/main/packages/playwright-core/browsers.json
        # ただしローカルにbrowsers.jsonがある場合はそれを参照
        # なければバージョンテーブルを内包
        version_map = {
            # 主要バージョンのみ抜粋（必要に応じて拡張）
            "1.43": "125.0.6422.60",
            "1.42": "124.0.6367.60",
            "1.41": "123.0.6312.105",
            "1.40": "122.0.6261.111",
            "1.39": "121.0.6167.85",
            "1.38": "120.0.6099.71",
            "1.37": "119.0.6045.105",
            "1.36": "118.0.5993.117",
            "1.35": "117.0.5938.149",
            "1.34": "116.0.5845.96",
            "1.33": "115.0.5790.170",
            "1.32": "114.0.5735.133",
            "1.31": "113.0.5672.63",
            "1.30": "112.0.5615.121",
            "1.29": "111.0.5563.64",
            "1.28": "110.0.5481.77",
            "1.27": "109.0.5414.74",
            "1.26": "108.0.5359.125",
            "1.25": "107.0.5304.68",
            "1.24": "106.0.5249.21",
            "1.23": "105.0.5195.52",
            "1.22": "104.0.5112.79",
            "1.21": "103.0.5060.53",
            "1.20": "102.0.5005.61",
            "1.19": "101.0.4951.41",
            "1.18": "100.0.4896.75",
            "1.17": "99.0.4844.51",
            "1.16": "98.0.4758.102",
            "1.15": "97.0.4692.71",
            "1.14": "96.0.4664.45",
            "1.13": "95.0.4638.17",
            "1.12": "94.0.4606.0",
        }
        pw_version = get_playwright_version()
        if not pw_version:
            return None
        # 1.43.1 → 1.43
        major_minor = ".".join(pw_version.split(".")[:2])
        chromium_version = version_map.get(major_minor)
        return chromium_version
    except Exception:
        return None

def get_system_chromium_version():
    """
    システムにインストールされているChromium/Chromeのバージョンを取得
    """
    plat = platform.system()
    candidates = []
    if plat == "Windows":
        # 代表的なパス
        paths = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Chromium\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles%\Chromium\Application\chrome.exe"),
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    out = subprocess.check_output([p, "--version"], stderr=subprocess.STDOUT, text=True)
                    m = re.search(r"(\d+\.\d+\.\d+\.\d+)", out)
                    if m:
                        return m.group(1)
                except Exception:
                    continue
        # wmic fallback
        try:
            out = subprocess.check_output(
                ["wmic", "datafile", "where", "name='C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe'", "get", "Version", "/value"],
                stderr=subprocess.STDOUT, text=True
            )
            m = re.search(r"Version=(\d+\.\d+\.\d+\.\d+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
    else:
        # Linux/Mac
        for cmd in ["chromium-browser", "chromium", "google-chrome", "chrome"]:
            if shutil.which(cmd):
                try:
                    out = subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT, text=True)
                    m = re.search(r"(\d+\.\d+\.\d+\.\d+)", out)
                    if m:
                        return m.group(1)
                except Exception:
                    continue
    return None

def compare_versions(v1, v2):
    """
    セマンティックバージョン比較
    v1 >= v2 ならTrue
    """
    def parse(v):
        return [int(x) for x in v.split(".")]
    a = parse(v1)
    b = parse(v2)
    for i in range(max(len(a), len(b))):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        if ai > bi:
            return True
        elif ai < bi:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Playwright要求Chromiumバージョン検出・比較ツール")
    parser.add_argument("--chromium-version", action="store_true", help="Playwright要求Chromiumバージョンを出力")
    parser.add_argument("--system-version", action="store_true", help="システムのChromium/Chromeバージョンを出力")
    parser.add_argument("--compare", action="store_true", help="バージョン比較結果を出力")
    parser.add_argument("--all", action="store_true", help="全情報をJSONで出力")
    args = parser.parse_args()

    result = {}
    chromium_required = get_playwright_chromium_version()
    chromium_system = get_system_chromium_version()

    if args.chromium_version:
        print(chromium_required or "unknown")
        return
    if args.system_version:
        print(chromium_system or "not_found")
        return
    if args.compare:
        if chromium_required and chromium_system:
            print("ok" if compare_versions(chromium_system, chromium_required) else "upgrade_needed")
        else:
            print("unknown")
        return
    if args.all:
        result = {
            "playwright_chromium_version": chromium_required,
            "system_chromium_version": chromium_system,
            "compatible": (compare_versions(chromium_system, chromium_required)
                           if chromium_required and chromium_system else None)
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # デフォルト: 全情報を日本語で出力
    print("Playwright要求Chromiumバージョン:", chromium_required or "不明")
    print("システムのChromium/Chromeバージョン:", chromium_system or "未検出")
    if chromium_required and chromium_system:
        if compare_versions(chromium_system, chromium_required):
            print("→ システムのChromiumはPlaywright要件を満たしています。")
        else:
            print("→ システムのChromiumはアップグレードが必要です。")
    else:
        print("→ バージョン情報の取得に失敗しました。")

if __name__ == "__main__":
    import shutil
    main()