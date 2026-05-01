FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/
# Install CPU-only PyTorch first (avoids pulling large CUDA binaries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install .

# Pre-download all NLP assets at build time so cold starts are fast
RUN python -m nltk.downloader -d /usr/share/nltk_data vader_lexicon punkt && \
    python -m spacy download en_core_web_md && \
    python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Pre-train risk model artifact
RUN python -c "from app.risk.train import train; train(save=True)"

ENV NLTK_DATA=/usr/share/nltk_data \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
