"""
Web Search Processing Module

Runs a query on one search engine (Google Custom Search, Brave, Exa, SearXNG) or,
in auto mode, on the configured engines in order until one answers.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..constants import SEARCH_ENGINES, SEARCH_GENRES
from .search_engines import (
    BraveSearchClient, CustomSearchAPIClient, ExaSearchClient, RateLimiter, SearXNGSearchClient,
)
from .search_engines.common import failure, main_result_type, parse_result_types
from .web_search_analysis import WebSearchAnalysisMixin

DEFAULT_AUTO_ORDER = ('google_custom', 'brave', 'exa', 'searxng')
ENGINE_ENV_VARS = {
    'google_custom': 'GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID',
    'brave': 'BRAVE_SEARCH_API_KEY',
    'exa': 'EXA_API_KEY',
    'searxng': 'SEARXNG_URL',
}
# Genres made of filetype: operators. Exa has no file type filter.
FILE_TYPE_GENRES = frozenset({'pdf', 'documents', 'presentations', 'spreadsheets'})
# Engines that can return result types other than web (Brave result sections).
NON_WEB_RESULT_ENGINES = frozenset({'brave'})


def resolve_default_engine() -> str:
    """SEARCH_ENGINE, else the legacy GOOGLE_SEARCH_MODE, else 'auto'."""
    engine = os.getenv('SEARCH_ENGINE', '').strip().lower()
    if engine:
        if engine in SEARCH_ENGINES or engine == 'auto':
            return engine
        logging.warning(f"Invalid SEARCH_ENGINE '{engine}'. Using 'auto'")
        return 'auto'

    legacy = os.getenv('GOOGLE_SEARCH_MODE', '').strip().lower()
    if legacy == 'custom_search_only':
        return 'google_custom'
    if legacy:
        logging.warning(
            f"GOOGLE_SEARCH_MODE='{legacy}' is no longer supported (googlesearch-python was removed). "
            "Using SEARCH_ENGINE=auto"
        )
    return 'auto'


def resolve_auto_order() -> Tuple[str, ...]:
    """SEARCH_ENGINE_ORDER (comma-separated) or DEFAULT_AUTO_ORDER."""
    raw = os.getenv('SEARCH_ENGINE_ORDER', '').strip()
    if not raw:
        return DEFAULT_AUTO_ORDER
    order: List[str] = []
    for name in (part.strip().lower() for part in raw.split(',')):
        if name in SEARCH_ENGINES and name not in order:
            order.append(name)
        elif name:
            logging.warning(f"Ignoring unknown engine '{name}' in SEARCH_ENGINE_ORDER")
    return tuple(order) or DEFAULT_AUTO_ORDER


class WebSearchProcessor(WebSearchAnalysisMixin):
    """Run web searches on Google Custom Search, Brave, Exa or SearXNG."""

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.clients = {
            'google_custom': CustomSearchAPIClient(),
            'brave': BraveSearchClient(),
            'exa': ExaSearchClient(),
            'searxng': SearXNGSearchClient(),
        }
        self.default_engine = resolve_default_engine()
        self.auto_order = resolve_auto_order()

        self.search_patterns = [
            r'site:([^\s]+)',
            r'filetype:([^\s]+)',
            r'"([^"]+)"',
            r'after:(\d{4})',
            r'before:(\d{4})'
        ]

    @property
    def custom_search_client(self) -> CustomSearchAPIClient:
        return self.clients['google_custom']

    def configured_engines(self) -> List[str]:
        return [name for name in SEARCH_ENGINES if self.clients[name].is_configured()]

    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate and analyze search query"""
        try:
            if not query or not query.strip():
                return {
                    'valid': False,
                    'error': 'Search query cannot be empty'
                }

            query = query.strip()
            if len(query) > 500:
                return {
                    'valid': False,
                    'error': 'Search query too long (max 500 characters)'
                }

            patterns_found = {}
            for pattern in self.search_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    pattern_name = pattern.split('(')[0].replace(':', '').replace('[', '').replace('^', '')
                    patterns_found[pattern_name] = matches

            return {
                'valid': True,
                'query': query,
                'length': len(query),
                'patterns': patterns_found,
                'is_advanced': len(patterns_found) > 0
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Query validation error: {str(e)}'
            }

    @staticmethod
    def _unsupported(name: str, genre: Optional[str], result_types: Tuple[str, ...]) -> Optional[Tuple[str, str, str]]:
        """(failure_kind, error_code, message) when the engine cannot serve the request."""
        non_web = [t for t in result_types if t != 'web']
        if non_web and name not in NON_WEB_RESULT_ENGINES:
            return ('unsupported_result_type', 'unsupported_result_type',
                    f"result_type {', '.join(non_web)} is not supported by {name} (brave only)")
        if name == 'exa' and genre in FILE_TYPE_GENRES:
            return ('unsupported_genre', 'unsupported_search_genre',
                    f"search_genre '{genre}' is not supported by exa (no file type filter)")
        return None

    def plan_engines(
        self, engine: Optional[str], search_genre: Optional[str], result_types: Tuple[str, ...] = ('web',)
    ) -> Tuple[List[str], List[Tuple[str, Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """
        Decide which engines to try.

        Returns (engines, skipped_attempts, error). error is set when nothing
        can run; skipped_attempts records engines auto mode left out.
        """
        requested = (engine or self.default_engine).strip().lower()
        genre = _normalize_genre(search_genre)

        if requested != 'auto':
            if requested not in SEARCH_ENGINES:
                return [], [], {
                    'success': False,
                    'error': f"Invalid engine: {requested!r}. Must be one of: auto, {', '.join(SEARCH_ENGINES)}",
                    'error_code': 'invalid_search_engine',
                }
            if not self.clients[requested].is_configured():
                return [], [], {
                    'success': False,
                    'error': f"Search engine '{requested}' is not configured",
                    'error_code': 'search_engine_not_configured',
                    'suggestion': f"Set {ENGINE_ENV_VARS[requested]}",
                    'configured_engines': self.configured_engines(),
                }
            unsupported = self._unsupported(requested, genre, result_types)
            if unsupported:
                _, code, message = unsupported
                suggestion = ('Use engine brave or google_custom, or omit search_genre'
                              if code == 'unsupported_search_genre'
                              else 'Use engine brave or auto, or result_type web')
                return [], [], {
                    'success': False,
                    'error': message,
                    'error_code': code,
                    'suggestion': suggestion,
                }
            return [requested], [], None

        engines: List[str] = []
        skipped: List[Tuple[str, Dict[str, Any]]] = []
        for name in self.auto_order:
            if not self.clients[name].is_configured():
                continue
            unsupported = self._unsupported(name, genre, result_types)
            if unsupported:
                # Engines that cannot return the result type at all are left out silently.
                if unsupported[0] != 'unsupported_result_type':
                    skipped.append((name, failure(unsupported[0], unsupported[2])))
                continue
            engines.append(name)

        if not engines:
            if result_types != ('web',):
                error = f"No configured engine supports result_type {', '.join(result_types)}"
            elif skipped:
                error = f"No configured engine supports search_genre '{genre}'"
            else:
                error = 'No search engine is configured'
            return [], skipped, {
                'success': False,
                'error': error,
                'error_code': 'search_engine_not_configured',
                'suggestion': 'Set one of: ' + '; '.join(ENGINE_ENV_VARS[n] for n in self.auto_order),
                'configured_engines': self.configured_engines(),
            }
        return engines, skipped, None

    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: str = 'en',
        region: str = 'us',
        safe_search: bool = True,
        search_genre: Optional[str] = None,
        recent_days: Optional[int] = None,
        engine: Optional[str] = None,
        result_type: Any = 'web',
    ) -> Dict[str, Any]:
        """
        Search with the requested engine, or the configured engines in auto order.

        result_type is a string ("web", "web,discussions") or a list; types
        other than web are Brave result sections (see parse_result_types).
        """
        try:
            validation = self.validate_query(query)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error'],
                    'query': query
                }

            num_results = max(1, min(100, num_results))
            result_types, type_error = parse_result_types(result_type)
            if type_error:
                return {'success': False, 'error': type_error, 'error_code': 'invalid_result_type', 'query': query}
            result_type = main_result_type(result_types)
            engines, attempts, error = self.plan_engines(engine, search_genre, result_types)
            if error:
                error['query'] = query
                return error

            result: Dict[str, Any] = {}
            for name in engines:
                if not self.rate_limiter.can_make_request(name):
                    wait = self.rate_limiter.get_wait_time(name)
                    result = failure('rate_limited', f'Client-side rate limit reached. Wait {wait:.1f} seconds')
                else:
                    try:
                        result = await self._search_engine(
                            name, query, num_results, language, region, safe_search, search_genre, recent_days,
                            result_types
                        )
                    except Exception as e:
                        result = failure('error', f'{name} search error: {e}')
                    status = 'success' if result['success'] else (
                        '429' if result.get('failure_kind') == 'rate_limited' else 'error')
                    self.rate_limiter.record_request(name, status)
                attempts.append((name, result))
                if result['success'] or result.get('failure_kind') == 'no_results':
                    break

            result = dict(result)
            result['engine'] = attempts[-1][0]
            result['result_type'] = result_type
            result.setdefault('query', query)
            if len(attempts) > 1:
                result['fallback_info'] = {
                    'attempts': [
                        {'engine': name, 'success': r['success'],
                         **({} if r['success'] else {'error': r.get('error'), 'failure_kind': r.get('failure_kind')})}
                        for name, r in attempts
                    ]
                }

            if result['success']:
                result['search_metadata'] = self._build_metadata(
                    result['results'], validation, num_results, language, region,
                    safe_search, search_genre, recent_days, result['engine'], result.get('enhanced_query', query)
                )
                result['search_metadata']['search_params']['result_type'] = list(result_types)
            elif result.get('failure_kind') == 'no_results':
                suggestions = self._generate_simplified_query_suggestions(query)
                if suggestions:
                    quoted = [f"'{s}'" for s in suggestions]
                    # Keep an engine's own hint (e.g. discussions only exist for some queries).
                    result.setdefault('suggestion', f"Try these simpler searches: {', '.join(quoted)}")
                    result['alternative_queries'] = suggestions
            elif len(engines) > 1:
                result['error'] = f"All search engines failed. Last error: {result.get('error')}"
            return result

        except Exception as e:
            return {
                'success': False,
                'error': f'Search processing error: {str(e)}',
                'query': query
            }

    async def search_google(
        self,
        query: str,
        num_results: int = 10,
        language: str = 'en',
        region: str = 'us',
        safe_search: bool = True,
        search_genre: Optional[str] = None,
        include_snippets: bool = True,
        recent_days: Optional[int] = None,
        engine: Optional[str] = None,
        result_type: Any = 'web',
    ) -> Dict[str, Any]:
        """Former GoogleSearchProcessor.search_google signature. include_snippets is ignored."""
        return await self.search(
            query, num_results, language, region, safe_search, search_genre, recent_days, engine, result_type
        )

    async def _search_engine(
        self,
        name: str,
        query: str,
        num_results: int,
        language: str,
        region: str,
        safe_search: bool,
        search_genre: Optional[str],
        recent_days: Optional[int],
        result_types: Tuple[str, ...] = ('web',),
    ) -> Dict[str, Any]:
        """Run one engine, applying search_genre the way that engine supports."""
        client = self.clients[name]
        if name == 'exa':
            if _normalize_genre(search_genre) == 'japanese':
                region = 'jp'
            return await client.search(query, num_results, region, safe_search, recent_days)

        enhanced_query = self._enhance_query_with_genre(query, search_genre)
        if name == 'brave':
            result = await client.search(enhanced_query, num_results, language, region, safe_search, recent_days,
                                         result_types)
        else:
            result = await client.search(enhanced_query, num_results, language, region, safe_search, recent_days)
        result['enhanced_query'] = enhanced_query
        return result

    def _build_metadata(
        self, results, validation, num_results, language, region,
        safe_search, search_genre, recent_days, engine, enhanced_query
    ) -> Dict[str, Any]:
        domains = [r['domain'] for r in results]
        type_counts: Dict[str, int] = {}
        for r in results:
            type_counts[r['type']] = type_counts.get(r['type'], 0) + 1
        return {
            'query_info': validation,
            'search_params': {
                'engine': engine,
                'num_results_requested': num_results,
                'language': language,
                'region': region,
                'safe_search': safe_search,
                'search_genre': search_genre,
                'recent_days': recent_days,
                'enhanced_query': enhanced_query,
            },
            'result_stats': {
                'total_results': len(results),
                'unique_domains': len(set(domains)),
                'domain_distribution': {d: domains.count(d) for d in set(domains)},
                'result_types': type_counts,
            },
        }

    def _enhance_query_with_genre(self, query: str, genre: Optional[str]) -> str:
        """Append the genre's search operator (see constants.SEARCH_GENRES)."""
        if not genre:
            return query

        spec = SEARCH_GENRES.get(str(genre).strip().lower())
        if spec:
            return f"{query} ({spec['operator']})"
        return query

    def get_available_genres(self) -> Dict[str, str]:
        """Get the supported search genres with descriptions."""
        return {name: spec['description'] for name, spec in SEARCH_GENRES.items()}


def _normalize_genre(genre: Optional[str]) -> Optional[str]:
    if not isinstance(genre, str) or not genre.strip():
        return None
    return genre.strip().lower()


# Backward-compatible name
GoogleSearchProcessor = WebSearchProcessor
