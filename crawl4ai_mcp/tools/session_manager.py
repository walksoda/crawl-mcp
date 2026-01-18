"""Session management for Crawl4AI MCP Server.

This module manages browser session persistence for web crawling,
storing and retrieving session data (cookies, localStorage) per domain.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


class SessionManager:
    """
    Manages browser session persistence for web crawling.

    Stores and retrieves session data (cookies, localStorage) per domain,
    enabling authenticated crawling without re-login on each request.

    Session data is stored in a JSON file (~/.crawl4ai_sessions.json) with
    secure file permissions (0600).

    Current Limitations:
    - Only user-provided cookies are saved (crawl4ai doesn't expose response cookies)
    - localStorage is stored but not applied (crawl_url doesn't support storage_state)
    - Full Playwright storage_state integration requires architectural changes

    Security Notes:
    - Session file is protected with owner-only permissions
    - URL credentials (user:pass@host) are stripped from domain keys
    - Consider additional encryption for sensitive environments
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize the session manager.

        Args:
            storage_path: Path to the session storage JSON file.
                         If None, uses a default path in the user's home directory.
        """
        if storage_path is None:
            home = Path.home()
            storage_path = str(home / ".crawl4ai_sessions.json")

        self.storage_path = storage_path
        self._sessions: Dict[str, dict] = {}
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load sessions from the storage file with validation."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate data structure - must be a dict
                    if not isinstance(data, dict):
                        print(f"Warning: Invalid session data format, expected dict")
                        self._sessions = {}
                        return

                    # Validate each domain entry
                    validated = {}
                    for domain, session in data.items():
                        if not isinstance(domain, str) or not isinstance(session, dict):
                            continue  # Skip invalid entries
                        # Ensure required fields exist with valid types
                        if 'storage_state' not in session:
                            continue
                        if not isinstance(session.get('storage_state'), dict):
                            continue
                        # Validate expires_at if present
                        expires_at = session.get('expires_at')
                        if expires_at is not None and not isinstance(expires_at, (int, float)):
                            session['expires_at'] = None  # Remove invalid expiry
                        validated[domain] = session

                    self._sessions = validated
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load sessions from {self.storage_path}: {e}")
                self._sessions = {}

    def _save_sessions(self) -> None:
        """Save sessions to the storage file with secure permissions."""
        import stat

        temp_path = self.storage_path + ".tmp"
        try:
            # Remove temp file if it exists to prevent symlink attacks
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

            # Create temp file with O_EXCL to prevent race conditions
            fd = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self._sessions, f, indent=2, ensure_ascii=False)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            # Atomic rename
            os.replace(temp_path, self.storage_path)

        except (IOError, TypeError, ValueError, OSError) as e:
            print(f"Warning: Could not save sessions to {self.storage_path}: {e}")
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _get_domain_key(self, url: str) -> str:
        """Extract domain from URL for session lookup, stripping credentials."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Strip user:pass@ from netloc for security
        netloc = parsed.netloc.lower()
        if '@' in netloc:
            netloc = netloc.split('@')[-1]
        return netloc

    def get_session(self, url: str) -> Optional[dict]:
        """
        Get stored session data for a URL's domain.

        Args:
            url: The URL to get session data for.

        Returns:
            Session data dict with 'cookies' and 'origins' keys, or None.
        """
        domain = self._get_domain_key(url)
        session = self._sessions.get(domain)

        if session and isinstance(session, dict):
            # Check if session has expired
            expires_at = session.get('expires_at')
            if isinstance(expires_at, (int, float)) and expires_at < time.time():
                # Session expired, remove it
                del self._sessions[domain]
                self._save_sessions()
                return None

            storage_state = session.get('storage_state')
            if isinstance(storage_state, dict):
                return storage_state

        return None

    def save_session(
        self,
        url: str,
        cookies: List[dict] = None,
        local_storage: Dict[str, str] = None,
        ttl_hours: int = 24
    ) -> None:
        """
        Save session data for a URL's domain.

        Args:
            url: The URL to save session data for.
            cookies: List of cookie dicts with name, value, domain, etc.
            local_storage: Dict of localStorage key-value pairs.
            ttl_hours: Time-to-live in hours for the session (default 24, min 1).
        """
        from urllib.parse import urlparse

        # Validate TTL
        if ttl_hours < 1:
            ttl_hours = 1  # Minimum 1 hour

        domain = self._get_domain_key(url)
        parsed = urlparse(url)
        # Strip credentials from origin for security
        netloc = parsed.netloc
        if '@' in netloc:
            netloc = netloc.split('@')[-1]
        origin = f"{parsed.scheme}://{netloc}"

        # Build storage_state in Playwright format
        storage_state = {
            "cookies": cookies or [],
            "origins": []
        }

        if local_storage:
            storage_state["origins"].append({
                "origin": origin,
                "localStorage": [
                    # Ensure values are strings for Playwright compatibility
                    {"name": str(k), "value": str(v)} for k, v in local_storage.items()
                ]
            })

        self._sessions[domain] = {
            "storage_state": storage_state,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl_hours * 3600),
            "domain": domain,
            "origin": origin
        }

        self._save_sessions()

    def save_storage_state(self, url: str, storage_state: dict, ttl_hours: int = 24) -> None:
        """
        Save a complete storage_state dict for a URL's domain.

        Args:
            url: The URL to save session data for.
            storage_state: Complete storage_state dict from Playwright.
            ttl_hours: Time-to-live in hours for the session (min 1).
        """
        # Validate TTL
        if not isinstance(ttl_hours, (int, float)) or ttl_hours < 1:
            ttl_hours = max(1, ttl_hours) if isinstance(ttl_hours, (int, float)) else 1

        domain = self._get_domain_key(url)

        self._sessions[domain] = {
            "storage_state": storage_state,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl_hours * 3600),
            "domain": domain
        }

        self._save_sessions()

    def clear_session(self, url: str) -> bool:
        """
        Clear session data for a URL's domain.

        Args:
            url: The URL to clear session data for.

        Returns:
            True if session was cleared, False if no session existed.
        """
        domain = self._get_domain_key(url)
        if domain in self._sessions:
            del self._sessions[domain]
            self._save_sessions()
            return True
        return False

    def clear_all_sessions(self) -> int:
        """
        Clear all stored sessions.

        Returns:
            Number of sessions cleared.
        """
        count = len(self._sessions)
        self._sessions = {}
        self._save_sessions()
        return count

    def list_sessions(self) -> List[dict]:
        """
        List all stored sessions with metadata.

        Returns:
            List of session info dicts with domain, created_at, expires_at.
        """
        current_time = time.time()
        result = []
        for domain, data in self._sessions.items():
            if not isinstance(data, dict):
                continue  # Skip invalid entries

            expires_at = data.get("expires_at")
            is_expired = False
            if isinstance(expires_at, (int, float)):
                is_expired = expires_at < current_time

            storage_state = data.get("storage_state", {})
            cookie_count = 0
            if isinstance(storage_state, dict):
                cookies = storage_state.get("cookies", [])
                if isinstance(cookies, list):
                    cookie_count = len(cookies)

            result.append({
                "domain": domain,
                "created_at": data.get("created_at"),
                "expires_at": expires_at,
                "is_expired": is_expired,
                "cookie_count": cookie_count
            })
        return result

    def get_browser_config_params(self, url: str) -> dict:
        """
        Get BrowserConfig parameters for session persistence.

        Args:
            url: The URL to get config for.

        Returns:
            Dict with storage_state and other browser config params.
        """
        session = self.get_session(url)
        if session:
            return {
                "storage_state": session,
                "use_persistent_context": True
            }
        return {}


# Global instance for module-level access
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global SessionManager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def extract_cookies_from_result(result) -> List[dict]:
    """
    Extract cookies from a crawl result.

    Note: crawl4ai doesn't expose response cookies in CrawlResponse,
    so this currently only works if cookies were explicitly set.

    Args:
        result: CrawlResponse object or dict

    Returns:
        List of cookie dicts
    """
    if hasattr(result, 'cookies'):
        return result.cookies or []
    elif isinstance(result, dict):
        return result.get('cookies', [])
    return []
