FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/
RUN pip install .

# fetch lightweight NLP assets at build time so cold start is fast
RUN python -m nltk.downloader -d /usr/share/nltk_data vader_lexicon punkt && \
    python -m spacy download en_core_web_md

# Pre-train risk model artifact
RUN python -c "from app.risk.train import train; train(save=True)"

ENV NLTK_DATA=/usr/share/nltk_data \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
