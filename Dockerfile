FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ORT_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--ws-max-size","65536"]
