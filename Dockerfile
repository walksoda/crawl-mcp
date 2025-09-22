# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for headless browsers and crawl4ai
RUN apt-get update && apt-get install -y \
    # Basic tools
    wget \
    curl \
    gnupg \
    ca-certificates \
    # Chrome/Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    # Additional browser dependencies
    fonts-liberation \
    libappindicator3-1 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libgtk-3-0 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libxtst6 \
    xdg-utils \
    # Video and audio codecs for rich content
    libavcodec-extra61 \
    libavformat-extra61 \
    # Webkit specific dependencies
    libgtk-4-1 \
    libxslt1.1 \
    libevent-2.1-7 \
    libwebpdemux2 \
    libenchant-2-2 \
    libsecret-1-0 \
    libhyphen0 \
    libgles2 \
    # Gstreamer for multimedia support
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    # Process management
    procps \
    # Build tools for Python wheels
    gcc \
    g++ \
    libicu-dev \
    libwoff1 \
    libharfbuzz-icu0 \
    libjpeg62-turbo \
    libwebp7 \
    libffi8 \
    libevdev2 \
    libjson-glib-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome for additional headless browser option
RUN wget -O /usr/share/keyrings/google-chrome.gpg https://dl.google.com/linux/linux_signing_key.pub \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright browsers (chromium, firefox, webkit)
RUN playwright install chromium firefox webkit

# Set up browser environment variables for headless operation
ENV DISPLAY=:99
ENV CHROME_BIN=/usr/bin/google-chrome
ENV CHROMIUM_BIN=/usr/bin/chromium-browser
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Configure headless browser settings for optimal performance
ENV CHROME_FLAGS="--no-sandbox --disable-dev-shm-usage --disable-gpu --disable-extensions --disable-default-apps --disable-translate --disable-device-discovery-notifications --disable-software-rasterizer --disable-background-timer-throttling --disable-backgrounding-occluded-windows --disable-renderer-backgrounding --disable-features=TranslateUI --disable-ipc-flooding-protection --disable-hang-monitor --disable-prompt-on-repost --no-first-run --no-default-browser-check --disable-logging --disable-permission-action-reporting"

# Copy the application code
COPY crawl4ai_mcp/ ./crawl4ai_mcp/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Set additional runtime environment variables for browser optimization
ENV PYTHONUNBUFFERED=1
ENV CRAWL4AI_BROWSER_TYPE=chromium
ENV CRAWL4AI_HEADLESS=true

# Expose port for HTTP mode (optional)
EXPOSE 8000

# Default command runs the MCP server
CMD ["python", "-m", "crawl4ai_mcp.server"]