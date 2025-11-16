# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire qa directory (build context is qa directory)
COPY . /app/

# Create a default config.yaml if it doesn't exist
RUN if [ ! -f /app/config.yaml ]; then \
    echo -e "retrieval:\n  dense:\n    top_k: 20\n    model_name: BAAI/bge-base-en-v1.5\nreranking:\n  enable: true\n  max_length: 512" > /app/config.yaml; \
    fi

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_DEBUG=false
ENV CONFIG_PATH=/app/config.yaml

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "qa_service.py"]

