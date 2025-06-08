"""
Additional extraction strategies and utilities for the Crawl4AI MCP server.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from crawl4ai.extraction_strategy import ExtractionStrategy


class CustomCssExtractionStrategy(ExtractionStrategy):
    """
    Custom CSS extraction strategy with enhanced features.
    """
    
    def __init__(self, selectors: Dict[str, str], flatten: bool = False):
        """
        Initialize the CSS extraction strategy.
        
        Args:
            selectors: Dictionary mapping field names to CSS selectors
            flatten: Whether to flatten nested results
        """
        self.selectors = selectors
        self.flatten = flatten
        super().__init__()
    
    async def extract(self, url: str, html: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Extract data using CSS selectors.
        
        Args:
            url: The URL being processed
            html: The HTML content
            
        Returns:
            Dictionary with extracted data
        """
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        results = {}
        
        for field_name, selector in self.selectors.items():
            elements = soup.select(selector)
            
            if len(elements) == 0:
                results[field_name] = None
            elif len(elements) == 1:
                results[field_name] = elements[0].get_text(strip=True)
            else:
                if self.flatten:
                    results[field_name] = [elem.get_text(strip=True) for elem in elements]
                else:
                    results[field_name] = [
                        {
                            'text': elem.get_text(strip=True),
                            'html': str(elem),
                            'attributes': elem.attrs
                        } for elem in elements
                    ]
        
        return results


class XPathExtractionStrategy(ExtractionStrategy):
    """
    XPath-based extraction strategy.
    """
    
    def __init__(self, xpath_expressions: Dict[str, str]):
        """
        Initialize the XPath extraction strategy.
        
        Args:
            xpath_expressions: Dictionary mapping field names to XPath expressions
        """
        self.xpath_expressions = xpath_expressions
        super().__init__()
    
    async def extract(self, url: str, html: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Extract data using XPath expressions.
        
        Args:
            url: The URL being processed
            html: The HTML content
            
        Returns:
            Dictionary with extracted data
        """
        from lxml import html as lxml_html
        
        tree = lxml_html.fromstring(html)
        results = {}
        
        for field_name, xpath in self.xpath_expressions.items():
            try:
                elements = tree.xpath(xpath)
                
                if not elements:
                    results[field_name] = None
                elif len(elements) == 1:
                    if hasattr(elements[0], 'text_content'):
                        results[field_name] = elements[0].text_content().strip()
                    else:
                        results[field_name] = str(elements[0]).strip()
                else:
                    results[field_name] = [
                        elem.text_content().strip() if hasattr(elem, 'text_content') 
                        else str(elem).strip() for elem in elements
                    ]
                    
            except Exception as e:
                results[field_name] = f"XPath error: {str(e)}"
        
        return results


class RegexExtractionStrategy(ExtractionStrategy):
    """
    Regular expression-based extraction strategy.
    """
    
    def __init__(self, patterns: Dict[str, str], flags: int = 0):
        """
        Initialize the regex extraction strategy.
        
        Args:
            patterns: Dictionary mapping field names to regex patterns
            flags: Regex flags to use
        """
        self.patterns = patterns
        self.flags = flags
        super().__init__()
    
    async def extract(self, url: str, html: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Extract data using regular expressions.
        
        Args:
            url: The URL being processed
            html: The HTML content
            
        Returns:
            Dictionary with extracted data
        """
        import re
        
        results = {}
        
        for field_name, pattern in self.patterns.items():
            try:
                matches = re.findall(pattern, html, self.flags)
                
                if not matches:
                    results[field_name] = None
                elif len(matches) == 1:
                    results[field_name] = matches[0]
                else:
                    results[field_name] = matches
                    
            except Exception as e:
                results[field_name] = f"Regex error: {str(e)}"
        
        return results


class SchemaValidationMixin:
    """
    Mixin class for validating extracted data against a schema.
    """
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data against a schema.
        
        Args:
            data: Extracted data
            schema: Validation schema
            
        Returns:
            Validated and cleaned data
        """
        validated_data = {}
        
        for field_name, field_schema in schema.items():
            if field_name in data:
                field_value = data[field_name]
                field_type = field_schema.get('type', 'string')
                
                try:
                    if field_type == 'integer':
                        validated_data[field_name] = int(field_value) if field_value else None
                    elif field_type == 'float':
                        validated_data[field_name] = float(field_value) if field_value else None
                    elif field_type == 'boolean':
                        validated_data[field_name] = bool(field_value) if field_value else None
                    elif field_type == 'list':
                        if isinstance(field_value, list):
                            validated_data[field_name] = field_value
                        else:
                            validated_data[field_name] = [field_value] if field_value else []
                    else:  # string
                        validated_data[field_name] = str(field_value) if field_value else None
                        
                except (ValueError, TypeError) as e:
                    validated_data[field_name] = f"Validation error: {str(e)}"
            else:
                # Field not found, check if required
                if field_schema.get('required', False):
                    validated_data[field_name] = "Required field missing"
                else:
                    validated_data[field_name] = field_schema.get('default')
        
        return validated_data


def create_extraction_strategy(
    strategy_type: str,
    config: Dict[str, Any]
) -> ExtractionStrategy:
    """
    Factory function to create extraction strategies.
    
    Args:
        strategy_type: Type of strategy ('css', 'xpath', 'regex', 'llm')
        config: Configuration for the strategy
        
    Returns:
        Configured extraction strategy
    """
    if strategy_type == "css":
        return CustomCssExtractionStrategy(
            selectors=config.get("selectors", {}),
            flatten=config.get("flatten", False)
        )
    elif strategy_type == "xpath":
        return XPathExtractionStrategy(
            xpath_expressions=config.get("expressions", {})
        )
    elif strategy_type == "regex":
        import re
        flags = 0
        if config.get("ignore_case", False):
            flags |= re.IGNORECASE
        if config.get("multiline", False):
            flags |= re.MULTILINE
        
        return RegexExtractionStrategy(
            patterns=config.get("patterns", {}),
            flags=flags
        )
    elif strategy_type == "llm":
        from crawl4ai.extraction_strategy import LLMExtractionStrategy
        return LLMExtractionStrategy(
            provider=config.get("provider", "openai"),
            api_token=config.get("api_token"),
            schema=config.get("schema", {}),
            extraction_type=config.get("extraction_type", "schema"),
            model=config.get("model", "gpt-3.5-turbo"),
        )
    else:
        raise ValueError(f"Unknown extraction strategy type: {strategy_type}")


# Export utility functions
__all__ = [
    'CustomCssExtractionStrategy',
    'XPathExtractionStrategy', 
    'RegexExtractionStrategy',
    'SchemaValidationMixin',
    'create_extraction_strategy'
]