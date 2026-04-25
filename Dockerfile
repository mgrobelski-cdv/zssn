FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential curl \
    software-properties-common git ffmpeg libsm6 libxext6 libgl1\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV BASE_URL="https://openrouter.ai/api/v1"
COPY streamlit_app.py .
COPY src/ ./src
COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]