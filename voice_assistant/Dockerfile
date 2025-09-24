FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY voice_assistant/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY voice_assistant ./voice_assistant

ENV PYTHONPATH=/app/voice_assistant/src:${PYTHONPATH}

EXPOSE 9000

CMD ["uvicorn", "voice_assistant.src.mcp_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "9000"]
