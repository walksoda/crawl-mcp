#!/usr/bin/env python3
"""
Crawl4AI MCP Server - DXT Entry Point

This is the main entry point for the DXT-packaged version of the Crawl4AI MCP server.
It handles configuration from the DXT environment and starts the FastMCP server.

Built by: walksoda (https://github.com/walksoda/crawl-mcp)
Powered by: unclecode's crawl4ai (https://github.com/unclecode/crawl4ai)
"""

import os
import sys
import logging
import platform
from pathlib import Path

# Add the server directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )

def install_dependencies():
    """Install required dependencies from requirements.txt"""
    import subprocess
    
    # Look for requirements.txt in parent directory (DXT package root)
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"Requirements file not found at {requirements_path}", file=sys.stderr)
        return False
    
    try:
        print(f"Installing dependencies from {requirements_path}...", file=sys.stderr)
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path),
            "--quiet", "--disable-pip-version-check"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Dependencies installed successfully", file=sys.stderr)
            
            # Additional check to ensure PDF dependencies are available
            try:
                import pdfminer
                print("PDF processing dependencies verified", file=sys.stderr)
            except ImportError:
                print("Installing additional PDF dependencies...", file=sys.stderr)
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "pdfminer-six>=20250506", "--quiet"
                ])
                
            return True
        else:
            print(f"pip install failed: {result.stderr}", file=sys.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during dependency installation: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point for DXT package"""
    try:
        # Setup logging from environment
        log_level = os.getenv('FASTMCP_LOG_LEVEL', 'INFO')
        setup_logging(log_level)
        logger = logging.getLogger(__name__)
        
        # Platform detection and logging
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }
        
        logger.info("Starting Crawl4AI MCP Server (DXT Package v1.0.7)")
        logger.info(f"Platform: {platform_info['system']} {platform_info['release']} ({platform_info['architecture']})")
        logger.info(f"Python version: {platform_info['python_version']}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Platform-specific information
        if platform_info['system'] == 'Linux':
            logger.info("Linux detected: Ensure system dependencies are installed (libnss3, libnspr4, etc.)")
        elif platform_info['system'] == 'Windows':
            logger.info("Windows detected: Visual C++ Build Tools may be required for some dependencies")
        elif platform_info['system'] == 'Darwin':
            logger.info(f"macOS detected: Architecture {platform_info['machine']}")
        
        # Log API key status (without revealing keys)
        api_keys_status = {
            'OpenAI': 'configured' if os.getenv('OPENAI_API_KEY') else 'not configured',
            'Anthropic': 'configured' if os.getenv('ANTHROPIC_API_KEY') else 'not configured',
            'Google': 'configured' if os.getenv('GOOGLE_API_KEY') else 'not configured'
        }
        logger.info(f"API Keys status: {api_keys_status}")
        
        # Import and run the MCP server
        try:
            from crawl4ai_mcp.server import main as server_main
            logger.info("Starting MCP server...")
            server_main()
        except ImportError as e:
            logger.error(f"Failed to import MCP server: {e}")
            logger.info("Attempting to install missing dependencies...")
            
            # Try to install dependencies automatically
            if install_dependencies():
                logger.info("Dependencies installed successfully. Retrying server start...")
                try:
                    from crawl4ai_mcp.server import main as server_main
                    logger.info("Starting MCP server...")
                    server_main()
                except Exception as retry_error:
                    logger.error(f"Server startup failed after dependency installation: {retry_error}")
                    sys.exit(1)
            else:
                logger.error("Failed to install dependencies automatically")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            logger.error("Check the logs above for more details")
            sys.exit(1)
            
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()