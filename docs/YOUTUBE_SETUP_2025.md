# 🎥 YouTube トランスクリプト抽出セットアップガイド（2025年最新版）

## 📋 概要

YouTube動画の字幕・トランスクリプト抽出には**youtube-transcript-api v1.1.0+**を使用します。
認証不要でシンプルかつ安定したトランスクリプト取得が可能です。

## 🔑 現在の設定状況

### ✅ 完了済み
- ✅ youtube-transcript-api v1.1.0+（インストール済み）
- ✅ YouTubeProcessor（アップデート済み）
- ✅ MCPサーバー統合（完了）
- ✅ トランスクリプト抽出機能（動作確認済み）

### ✨ 主な特徴
- ✅ 認証不要（APIキーやOAuth不要）
- ✅ 高速で安定したトランスクリプト取得
- ✅ 多言語対応（自動生成・手動字幕両対応）
- ✅ タイムスタンプ付きセグメント情報

## 🚀 使用方法

### 直接使用例

```python
from youtube_transcript_api import YouTubeTranscriptApi

# 基本的な使用方法
transcript = YouTubeTranscriptApi.get_transcript('VIDEO_ID', languages=['ja', 'en'])

# 利用可能な字幕を確認
transcript_list = YouTubeTranscriptApi.list_transcripts('VIDEO_ID')
for transcript in transcript_list:
    print(f"{transcript.language} ({transcript.language_code})")
```

### MCPサーバー経由

```python
# MCPサーバーの関数を使用
from crawl4ai_mcp.server import extract_youtube_transcript

result = await extract_youtube_transcript(
    url='https://www.youtube.com/watch?v=VIDEO_ID',
    languages=['ja', 'en'],
    include_timestamps=True
)
```

## 🔧 テスト方法

### 1. 直接テスト
```bash
python test_youtube_transcript_direct.py "https://www.youtube.com/watch?v=UJnPNIoeqzI"
```

### 2. MCPサーバーテスト
```bash
python test_mcp_youtube.py
```

## 📝 対応機能

### 🎯 基本機能
- ✅ 動画ID抽出（YouTube URL対応）
- ✅ 利用可能な字幕言語取得
- ✅ 自動生成・手動字幕の判別
- ✅ トランスクリプト取得（タイムスタンプ付き）
- ✅ 多言語優先順位対応

### 🌟 高度な機能
- ✅ バッチ処理（複数動画の同時処理）
- ✅ エラーハンドリング（詳細なエラーメッセージ）
- ✅ フォーマット対応（プレーンテキスト・タイムスタンプ付き）
- ✅ 翻訳機能（利用可能な場合）

## 🚫 制限事項

### ❌ 利用できない場合
- 字幕が無効化されている動画
- プライベート動画
- 地域制限のある動画
- 削除された動画

### ⚠️ 注意事項
- YouTube側の仕様変更により一時的に動作しない場合があります
- 大量のリクエストは避けてください（レート制限）
- 商用利用時はYouTubeの利用規約を確認してください

## 🆚 YouTube Data API v3 との比較

| 機能 | youtube-transcript-api | YouTube Data API v3 |
|------|----------------------|-------------------|
| 認証 | ❌ 不要 | ✅ 必要（OAuth/APIキー） |
| セットアップ | 🟢 簡単 | 🔴 複雑 |
| 字幕取得 | ✅ 対応 | ✅ 対応 |
| 動画メタデータ | ❌ 制限的 | ✅ 完全 |
| レート制限 | 🟡 緩い | 🔴 厳しい |
| 安定性 | 🟢 高い | 🟡 中程度 |

## 🎉 結論

**youtube-transcript-api v1.1.0+** への移行により：
- ✅ セットアップが大幅に簡素化
- ✅ 認証不要で即座に利用可能
- ✅ 安定したトランスクリプト取得
- ✅ メンテナンス負荷の軽減

これで YouTube トランスクリプト抽出機能の準備が完了しました！