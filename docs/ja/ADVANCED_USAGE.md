# 高度な使用ガイド

Crawl4AI MCPサーバーのパワーユーザー向けの高度なパターン、技術、ワークフローです。

## 🚁 高度なWebクローリング技術

### 深度サイト探索

**戦略的フィルタリング付きマルチ深度クローリング:**
```json
{
  "url": "https://docs.example.com",
  "max_depth": 3,
  "crawl_strategy": "best_first",
  "url_pattern": "*docs*",
  "score_threshold": 0.7,
  "content_filter": "bm25",
  "filter_query": "API ドキュメント チュートリアル ガイド"
}
```

**JavaScript重要サイトの最適化:**
```json
{
  "url": "https://spa-application.com",
  "wait_for_js": true,
  "simulate_user": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded",
  "execute_js": "window.scrollTo(0, document.body.scrollHeight); await new Promise(r => setTimeout(r, 2000));",
  "headers": {
    "User-Agent": "Mozilla/5.0 (compatible; CrawlBot/1.0)"
  }
}
```

### 困難サイト向けフォールバック戦略

**自動フォールバック付きマルチ戦略クローリング:**
```json
{
  "url": "https://protected-site.com",
  "cookies": {
    "session_id": "your_session_cookie",
    "auth_token": "bearer_token"
  },
  "headers": {
    "Authorization": "Bearer your-api-key",
    "Accept": "text/html,application/xhtml+xml"
  },
  "user_agent": "CustomBot/1.0",
  "simulate_user": true,
  "timeout": 90
}
```

**最大信頼性のためのcrawl_url_with_fallback使用:**
- 複数戦略の自動試行
- アンチボット保護への対応
- 詳細な失敗分析
- タイムアウト時の部分結果返却

## 🧠 AI搭載コンテンツ処理

### インテリジェントコンテンツ抽出

**カスタム指示付き意味的理解:**
```json
{
  "url": "https://research-paper.com",
  "extraction_goal": "学術論文から手法、結果、結論を抽出",
  "content_filter": "llm",
  "custom_instructions": "定量的結果、統計的有意性、実用的含意に焦点を当てる。参考文献と著者情報は無視する。",
  "use_llm": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4"
}
```

**マルチプロバイダーLLM設定:**
```json
{
  "mcpServers": {
    "crawl4ai-multi-llm": {
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "AZURE_OPENAI_API_KEY": "your-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com"
      }
    }
  }
}
```

### 大容量コンテンツの自動要約

**異なるコンテンツタイプ向け設定:**
```json
{
  "auto_summarize": true,
  "max_content_tokens": 50000,
  "summary_length": "long",
  "llm_provider": "anthropic",
  "chunk_content": true,
  "chunk_strategy": "topic"
}
```

**要約長の最適化:**
- `"short"`: 1-2段落（500-1000トークン）
- `"medium"`: 3-5段落（1500-3000トークン）
- `"long"`: 包括的分析（5000-10000トークン）

## 🔄 バッチ処理操作

### 大規模クローリングワークフロー

**URL管理付きバッチクローリング:**
```json
{
  "urls": [
    "https://site1.com/page1",
    "https://site1.com/page2",
    "https://site2.com/api-docs",
    "https://site3.com/tutorials"
  ],
  "config": {
    "generate_markdown": true,
    "extract_media": false,
    "timeout": 45,
    "auto_summarize": true
  },
  "base_timeout": 60
}
```

**YouTubeトランスクリプトバッチ処理:**
```json
{
  "request": {
    "urls": [
      "https://youtube.com/watch?v=VIDEO1",
      "https://youtube.com/watch?v=VIDEO2",
      "https://youtube.com/watch?v=VIDEO3"
    ],
    "languages": ["ja", "en"],
    "include_timestamps": true,
    "max_concurrent": 3,
    "translate_to": "ja"
  }
}
```

### 検索とクロールワークフロー

**競合分析パイプライン:**
```json
{
  "search_query": "企業API ドキュメント ベストプラクティス 2024",
  "num_search_results": 10,
  "crawl_top_results": 5,
  "search_genre": "technical",
  "extract_media": false,
  "generate_markdown": true
}
```

**コンテンツ研究ワークフロー:**
```json
{
  "request": {
    "queries": [
      "機械学習トレンド 2024",
      "AI導入 企業調査",
      "深層学習フレームワーク比較"
    ],
    "num_results_per_query": 15,
    "search_genre": "academic",
    "max_concurrent": 2
  }
}
```

## 🎯 複雑なワークフローパターン

### マルチステージコンテンツ分析

**1. 発見フェーズ:**
```bash
# 関連コンテンツを検索
search_google: "トピック キーワード 研究論文"
```

**2. 収集フェーズ:**
```bash
# フォールバック保護付きで上位結果をクロール
search_and_crawl: 上位5結果 → 包括的分析生成
```

**3. 分析フェーズ:**
```bash
# 構造化洞察を抽出
extract_structured_data: "手法、発見、含意"
```

**4. 統合フェーズ:**
```bash
# 関連コンテンツをバッチ処理
batch_crawl: 関連URL → 自動要約 → 比較分析
```

### サイトドキュメントマッピング

**完全なドキュメント抽出:**
```json
{
  "url": "https://api-docs.example.com",
  "max_depth": 4,
  "crawl_strategy": "bfs",
  "url_pattern": "*docs*,*api*,*guide*",
  "extract_media": true,
  "content_filter": "bm25",
  "filter_query": "API エンドポイント パラメータ 例 チュートリアル",
  "auto_summarize": false,
  "generate_markdown": true
}
```

### エンティティ抽出ワークフロー

**連絡先情報マイニング:**
```json
{
  "url": "https://company-directory.com",
  "entity_types": ["emails", "phones", "social_media"],
  "deduplicate": true,
  "include_context": true,
  "custom_patterns": {
    "linkedin": "linkedin\\.com/in/[\\w-]+",
    "github": "github\\.com/[\\w-]+",
    "departments": "部署:\\s*([\\w\\s]+)"
  }
}
```

## 🔧 パフォーマンス最適化

### メモリと処理管理

**大容量ドキュメント処理:**
```json
{
  "chunk_content": true,
  "chunk_size": 8000,
  "chunk_strategy": "topic",
  "overlap_rate": 0.1,
  "max_content_tokens": 100000,
  "auto_summarize": true
}
```

**タイムアウト最適化戦略:**
```json
{
  "timeout": 120,
  "base_timeout": 45,
  "wait_for_js": true,
  "simulate_user": false,
  "cache_mode": "enabled"
}
```

### 並行処理制御

**バッチ操作制限:**
- `max_concurrent`: 3（デフォルト、安定）
- `max_concurrent`: 5（最大、安定性に影響の可能性）
- `max_concurrent`: 1（順次、最も信頼性が高い）

**レート制限の考慮事項:**
- YouTube API: 3並行リクエスト
- Google検索: 2並行クエリ
- 一般クローリング: 5並行URL

## 🛡️ エラー処理と復元力

### 堅牢な設定パターン

**エラー回復付き本番対応セットアップ:**
```json
{
  "mcpServers": {
    "crawl4ai-production": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "ja",
        "FASTMCP_LOG_LEVEL": "ERROR",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**フォールバック戦略実装:**
1. **主要試行**: 標準crawl_url
2. **フォールバック1**: crawl_url_with_fallback
3. **フォールバック2**: 簡略化パラメータ（JS無し、基本抽出）
4. **最終フォールバック**: 手動URL検証と基本HTTP取得

### 監視と診断

**ヘルスチェック設定:**
```json
{
  "url": "http://127.0.0.1:8000/health",
  "timeout": 5000,
  "expected_status": 200,
  "check_interval": 30000
}
```

**トラブルシューティング用デバッグ設定:**
```json
{
  "env": {
    "FASTMCP_LOG_LEVEL": "DEBUG",
    "DEBUG": "1",
    "PYTHONUNBUFFERED": "1",
    "PLAYWRIGHT_DEBUG": "1"
  }
}
```

## 🎨 高度なコンテンツフィルタリング

### マルチステージフィルタリングパイプライン

**BM25 + LLM組み合わせ:**
```json
{
  "content_filter": "bm25",
  "filter_query": "技術ドキュメント APIリファレンス",
  "use_llm": true,
  "custom_instructions": "BM25フィルタリング後、意味分析を適用して実用的な技術情報のみを抽出"
}
```

**階層的コンテンツ処理:**
```json
{
  "filtering_strategy": "llm",
  "extract_top_chunks": 15,
  "similarity_threshold": 0.8,
  "chunking_strategy": "topic",
  "merge_strategy": "hierarchical"
}
```

## 📊 分析とレポーティング

### 高度なメトリクス収集

**パフォーマンス追跡:**
- URL当たりの処理時間
- 成功/失敗率
- 操作別トークン使用量
- キャッシュヒット率
- メモリ使用パターン

**コンテンツ品質メトリクス:**
- 抽出完全性
- 関連性スコアリング
- 重複検出率
- フォーマット変換成功率

### カスタムレポーティングワークフロー

**包括サイト分析:**
```bash
1. deep_crawl_site → 構造マッピング
2. extract_structured_data → コンテンツカテゴリ化
3. batch_crawl → 複数ページ抽出
4. multi_url_crawl → 比較分析
5. generate_report → 構造化出力
```

## 🔗 統合パターン

### API統合例

**エラー処理付きPython統合:**
```python
import asyncio
import logging
from typing import Optional, Dict, Any

async def robust_crawl_workflow(url: str, max_retries: int = 3) -> Optional[Dict[Any, Any]]:
    """自動フォールバックとリトライロジック付き高度クローリング"""
    
    strategies = [
        {"wait_for_js": True, "timeout": 60},
        {"wait_for_js": False, "timeout": 30},
        {"simulate_user": False, "timeout": 15}
    ]
    
    for attempt, strategy in enumerate(strategies):
        try:
            result = await crawl_url(url, **strategy)
            if result.get("success"):
                return result
        except Exception as e:
            logging.warning(f"試行 {attempt + 1} 失敗: {e}")
            if attempt < len(strategies) - 1:
                await asyncio.sleep(2 ** attempt)  # 指数バックオフ
            
    # crawl_url_with_fallbackを使用した最終フォールバック
    try:
        return await crawl_url_with_fallback(url)
    except Exception as e:
        logging.error(f"{url}のすべてのクローリング戦略が失敗: {e}")
        return None
```

**Node.jsバッチ処理:**
```javascript
async function processUrlBatch(urls, options = {}) {
  const batchSize = options.batchSize || 5;
  const results = [];
  
  for (let i = 0; i < urls.length; i += batchSize) {
    const batch = urls.slice(i, i + batchSize);
    
    try {
      const batchResults = await Promise.allSettled(
        batch.map(url => crawlUrl(url, options))
      );
      
      results.push(...batchResults);
      
      // レート制限遅延
      if (i + batchSize < urls.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (error) {
      console.error(`バッチ処理エラー:`, error);
    }
  }
  
  return results;
}
```

## 🚨 高度なトラブルシューティング

### 複雑な問題解決

**大規模操作のメモリ管理:**
```json
{
  "chunk_content": true,
  "max_content_tokens": 25000,
  "auto_summarize": true,
  "cache_mode": "bypass",
  "timeout": 300
}
```

**ネットワークと接続の問題:**
```json
{
  "user_agent": "Mozilla/5.0 (compatible; Bot/1.0)",
  "headers": {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
  },
  "cookies": {},
  "simulate_user": true,
  "timeout": 90
}
```

### パフォーマンスプロファイリング

**ボトルネックの特定:**
1. DEBUGログを有効化
2. トークン使用パターンを監視
3. 操作別処理時間を追跡
4. バッチ操作中のメモリ使用量を分析
5. ネットワークリクエストパターンをプロファイル

## 📚 関連ドキュメント

- **設定例**: [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
- **APIリファレンス**: [API_REFERENCE.md](API_REFERENCE.md)
- **HTTP統合**: [HTTP_INTEGRATION.md](HTTP_INTEGRATION.md)
- **開発ガイド**: [DEVELOPMENT.md](DEVELOPMENT.md)
- **インストールガイド**: [INSTALLATION.md](INSTALLATION.md)

## 💡 プロのコツ

1. **シンプルから始めて複雑さを段階的に追加** - 基本クローリングから始め、機能を段階的に追加
2. **リソース使用量を監視** - 最適化のためメモリと処理時間を追跡
3. **適切なツールを使用** - タスク要件にツールの複雑さを合わせる
4. **フォールバックを実装** - 重要なワークフローには常にバックアップ戦略を用意
5. **戦略的にキャッシュ** - パフォーマンスとコンテンツの新鮮さのバランスを保つ
6. **十分にテスト** - 開発環境で複雑なワークフローを検証
7. **パターンを文書化** - 成功した設定を再利用のために記録
8. **API制限を監視** - 外部サービスのレート制限とクォータを尊重