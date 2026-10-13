FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    bzip2 \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create user
RUN useradd -m -u 1000 user
RUN mkdir -p /app/models && chown -R user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy Code
COPY --chown=user:user . .

# Force fresh build for models (Cache Buster)
ENV REFRESHED_AT=2025-01-27_1730

# Run setup script to download models
RUN chmod +x setup_models.sh && ./setup_models.sh

# Verify download
RUN echo "Verifying model files..." && ls -R models/

# Fix metadata and tokens (Run during build to bake into image)
# 1. slice_joiner.py -> Cuts model from 8198 to 8193
# 2. fix_metadata.py -> Injects missing TDT fields
# 3. fix_tokens.py -> Generates matching token list
RUN echo "Running fix scripts..." \
    && python slice_joiner.py \
    && python fix_metadata.py \
    && python fix_tokens.py

EXPOSE 7860

# Run Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
