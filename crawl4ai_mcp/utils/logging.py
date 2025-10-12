"""
Logging utilities for Crawl4AI MCP Server.

Provides configurable logging to console and optional file output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union


class MCPLogger:
    """Enhanced logger for MCP server with console and file support."""

    def __init__(
        self,
        name: str = "crawl4ai_mcp",
        level: Union[str, int] = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_string: Optional[str] = None
    ):
        """
        Initialize the MCP logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for file logging
            max_file_size: Maximum file size before rotation (bytes)
            backup_count: Number of backup files to keep
            format_string: Custom format string (uses default if None)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._parse_level(level))

        # Remove any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Default format
        if format_string is None:
            format_string = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"

        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler (always enabled)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            # Ensure directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent duplicate messages from parent loggers
        self.logger.propagate = False

    def _parse_level(self, level: Union[str, int]) -> int:
        """Parse logging level from string or int."""
        if isinstance(level, str):
            level = level.upper()
            return getattr(logging, level, logging.INFO)
        return level

    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)

    def set_level(self, level: Union[str, int]):
        """Change logging level."""
        self.logger.setLevel(self._parse_level(level))

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


# Global logger instance
_default_logger: Optional[MCPLogger] = None


def get_logger(
    name: str = "crawl4ai_mcp",
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> MCPLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        max_file_size: Max file size before rotation
        backup_count: Number of backup files
        format_string: Custom format string

    Returns:
        MCPLogger instance
    """
    global _default_logger

    if _default_logger is None or _default_logger.logger.name != name:
        _default_logger = MCPLogger(
            name=name,
            level=level,
            log_file=log_file,
            max_file_size=max_file_size,
            backup_count=backup_count,
            format_string=format_string
        )

    return _default_logger


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> MCPLogger:
    """
    Set up the default logger for the application.

    This is a convenience function that creates a logger with the default name.

    Args:
        level: Logging level
        log_file: Optional file path for logging
        max_file_size: Max file size before rotation
        backup_count: Number of backup files
        format_string: Custom format string

    Returns:
        Configured MCPLogger instance
    """
    return get_logger(
        name="crawl4ai_mcp",
        level=level,
        log_file=log_file,
        max_file_size=max_file_size,
        backup_count=backup_count,
        format_string=format_string
    )


# Convenience functions for quick logging
def debug(message: str, *args, **kwargs):
    """Log debug message to default logger."""
    get_logger().debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """Log info message to default logger."""
    get_logger().info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """Log warning message to default logger."""
    get_logger().warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """Log error message to default logger."""
    get_logger().error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """Log critical message to default logger."""
    get_logger().critical(message, *args, **kwargs)


def exception(message: str, *args, **kwargs):
    """Log exception to default logger."""
    get_logger().exception(message, *args, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test console-only logging
    logger = setup_logging(level="DEBUG")
    logger.info("Console logging test")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test with file logging
    logger_file = setup_logging(
        level="INFO",
        log_file="test.log",
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_file.info("File logging test")
    logger_file.error("Error written to file")

    print("Logging test completed. Check test.log for file output.")
