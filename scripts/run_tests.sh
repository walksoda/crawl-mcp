#!/bin/bash
# MCP Server Test Runner
# Usage: ./scripts/run_tests.sh [mode] [options]
#
# Modes:
#   quick   - Run quick unit tests only
#   unit    - Run all unit tests
#   mcp     - Run MCP tool tests (excludes slow tests)
#   full    - Run all tests including slow tests
#   youtube - Run YouTube tool tests only
#   crawl   - Run crawl tool tests only
#   search  - Run search tool tests only
#   file    - Run file tool tests only
#   utility - Run utility tool tests only
#
# Options:
#   cov     - Enable coverage reporting
#   verbose - Show verbose output
#   debug   - Show debug output

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
MODE="${1:-quick}"
shift || true

COVERAGE=""
VERBOSE="-v"
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        cov|coverage)
            COVERAGE="--cov=crawl4ai_mcp --cov-report=term-missing --cov-report=html"
            ;;
        verbose)
            VERBOSE="-vv"
            ;;
        debug)
            VERBOSE="-vvs"
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# Activate venv if exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Check pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest not found. Install with: pip install pytest pytest-asyncio${NC}"
    exit 1
fi

echo -e "${BLUE}Running tests in mode: ${YELLOW}$MODE${NC}"
echo ""

case $MODE in
    quick)
        echo -e "${GREEN}Running quick unit tests...${NC}"
        pytest tests/unit/ $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    unit)
        echo -e "${GREEN}Running all unit tests...${NC}"
        pytest tests/unit/ $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    mcp)
        echo -e "${GREEN}Running MCP tool tests (excluding slow)...${NC}"
        pytest tests/mcp/ $VERBOSE --tb=short -m "not slow" $COVERAGE $EXTRA_ARGS
        ;;
    full)
        echo -e "${GREEN}Running all tests (including slow)...${NC}"
        pytest tests/unit/ tests/mcp/ $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    youtube)
        echo -e "${GREEN}Running YouTube tool tests...${NC}"
        pytest tests/mcp/test_youtube_tools.py $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    crawl)
        echo -e "${GREEN}Running crawl tool tests...${NC}"
        pytest tests/mcp/test_crawl_tools.py $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    search)
        echo -e "${GREEN}Running search tool tests...${NC}"
        pytest tests/mcp/test_search_tools.py $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    file)
        echo -e "${GREEN}Running file tool tests...${NC}"
        pytest tests/mcp/test_file_tools.py $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    utility)
        echo -e "${GREEN}Running utility tool tests...${NC}"
        pytest tests/mcp/test_utility_tools.py $VERBOSE --tb=short $COVERAGE $EXTRA_ARGS
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo ""
        echo "Available modes:"
        echo "  quick   - Run quick unit tests only"
        echo "  unit    - Run all unit tests"
        echo "  mcp     - Run MCP tool tests (excludes slow tests)"
        echo "  full    - Run all tests including slow tests"
        echo "  youtube - Run YouTube tool tests only"
        echo "  crawl   - Run crawl tool tests only"
        echo "  search  - Run search tool tests only"
        echo "  file    - Run file tool tests only"
        echo "  utility - Run utility tool tests only"
        echo ""
        echo "Options:"
        echo "  cov     - Enable coverage reporting"
        echo "  verbose - Show verbose output"
        echo "  debug   - Show debug output with print statements"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Tests completed!${NC}"
