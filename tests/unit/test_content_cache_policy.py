"""Unit tests for ContentCachePolicy."""

import json
import os
import threading
import time

import pytest

from crawl4ai_mcp.infra.content_cache_policy import ContentCachePolicy


@pytest.fixture
def policy(tmp_path):
    """Create a ContentCachePolicy with a temporary storage path."""
    path = str(tmp_path / "freshness.json")
    return ContentCachePolicy(storage_path=path, ttl_seconds=3600)


class TestResolveCacheMode:
    """Tests for resolve_cache_mode."""

    def test_disabled_when_caching_off(self, policy):
        result = policy.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="enabled", enable_caching=False,
        )
        assert result == "disabled"

    def test_disabled_when_explicitly_disabled(self, policy):
        result = policy.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="disabled", enable_caching=True,
        )
        assert result == "disabled"

    def test_bypass_when_explicitly_bypass(self, policy):
        result = policy.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="bypass", enable_caching=True,
        )
        assert result == "bypass"

    def test_enabled_when_content_offset_positive(self, policy):
        result = policy.resolve_cache_mode(
            url="https://example.com", content_offset=5000,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "enabled"

    def test_enabled_when_ttl_zero(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=0)
        result = p.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "enabled"

    def test_bypass_when_url_not_recorded(self, policy):
        result = policy.resolve_cache_mode(
            url="https://new-site.com", content_offset=0,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "bypass"

    def test_enabled_when_within_ttl(self, policy):
        policy.record_fresh_fetch("https://example.com")
        result = policy.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "enabled"

    def test_bypass_when_ttl_expired(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=1)
        p.record_fresh_fetch("https://example.com")
        # Manually expire the entry
        p._entries["https://example.com"] = time.time() - 10
        result = p.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "bypass"

    def test_pagination_overrides_expired_ttl(self, tmp_path):
        """content_offset > 0 forces enabled even if TTL is expired."""
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=1)
        # No record -> would normally be bypass
        result = p.resolve_cache_mode(
            url="https://example.com", content_offset=5000,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "enabled"


class TestRecordFreshFetch:
    """Tests for record_fresh_fetch."""

    def test_records_timestamp(self, policy):
        before = time.time()
        policy.record_fresh_fetch("https://example.com")
        after = time.time()
        ts = policy._entries["https://example.com"]
        assert before <= ts <= after

    def test_updates_existing_timestamp(self, policy):
        policy.record_fresh_fetch("https://example.com")
        old_ts = policy._entries["https://example.com"]
        time.sleep(0.01)
        policy.record_fresh_fetch("https://example.com")
        new_ts = policy._entries["https://example.com"]
        assert new_ts > old_ts

    def test_after_record_resolve_returns_enabled(self, policy):
        policy.record_fresh_fetch("https://example.com")
        result = policy.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "enabled"


class TestPersistence:
    """Tests for JSON persistence."""

    def test_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        p1 = ContentCachePolicy(storage_path=path, ttl_seconds=3600)
        p1.record_fresh_fetch("https://example.com")

        # Load a new instance from the same file
        p2 = ContentCachePolicy(storage_path=path, ttl_seconds=3600)
        assert "https://example.com" in p2._entries

    def test_recovers_from_corrupted_json(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        with open(path, "w") as f:
            f.write("{invalid json!!!")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=3600)
        assert p._entries == {}

    def test_recovers_from_non_dict_json(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        p = ContentCachePolicy(storage_path=path, ttl_seconds=3600)
        assert p._entries == {}

    def test_filters_invalid_entries(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        with open(path, "w") as f:
            json.dump({
                "https://good.com": 1234567890.0,
                "https://bad.com": "not_a_number",
                123: 1234567890.0,
            }, f)
        p = ContentCachePolicy(storage_path=path, ttl_seconds=3600)
        assert "https://good.com" in p._entries
        assert "https://bad.com" not in p._entries


class TestGarbageCollection:
    """Tests for GC during save."""

    def test_expired_entries_removed_on_save(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=60)
        # Add an expired entry directly
        p._entries["https://expired.com"] = time.time() - 120
        p._entries["https://fresh.com"] = time.time()
        p._save()
        assert "https://expired.com" not in p._entries
        assert "https://fresh.com" in p._entries

    def test_gc_disabled_when_ttl_zero(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=0)
        old_ts = time.time() - 999999
        p._entries["https://old.com"] = old_ts
        p._save()
        # Should not be removed when TTL is 0
        assert "https://old.com" in p._entries


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_record_and_resolve(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=3600)
        errors = []

        def writer(url_idx):
            try:
                for _ in range(20):
                    p.record_fresh_fetch(f"https://site-{url_idx}.com")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(20):
                    p.resolve_cache_mode(
                        url=f"https://site-{i % 5}.com",
                        content_offset=0,
                        requested_mode="enabled",
                        enable_caching=True,
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for _ in range(3):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for t in threads:
            assert not t.is_alive(), f"Thread {t.name} still alive - possible deadlock"
        assert errors == [], f"Thread errors: {errors}"


class TestEnvironmentVariable:
    """Tests for CRAWL4AI_CACHE_TTL_SECONDS env var."""

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_CACHE_TTL_SECONDS", "7200")
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path)
        assert p.ttl_seconds == 7200

    def test_env_var_zero_disables_ttl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_CACHE_TTL_SECONDS", "0")
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path)
        assert p.ttl_seconds == 0

    def test_invalid_env_var_uses_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_CACHE_TTL_SECONDS", "not_a_number")
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path)
        assert p.ttl_seconds == 3600


class TestNegativeTTL:
    """Tests for negative TTL clamping."""

    def test_negative_ttl_clamped_to_zero(self, tmp_path):
        """Negative TTL is treated as 0 (infinite cache)."""
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=-100)
        assert p.ttl_seconds == 0
        # Should behave as infinite cache (enabled, not bypass)
        result = p.resolve_cache_mode(
            url="https://example.com", content_offset=0,
            requested_mode="enabled", enable_caching=True,
        )
        assert result == "enabled"

    def test_negative_one_clamped_to_zero(self, tmp_path):
        path = str(tmp_path / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=-1)
        assert p.ttl_seconds == 0


class TestMissingParentDirectory:
    """Tests for storage path with missing parent directories."""

    def test_missing_parent_directory(self, tmp_path):
        """record_fresh_fetch should not raise when parent dir does not exist."""
        path = str(tmp_path / "nonexistent" / "subdir" / "freshness.json")
        p = ContentCachePolicy(storage_path=path, ttl_seconds=3600)
        p.record_fresh_fetch("https://example.com")
        assert "https://example.com" in p._entries
