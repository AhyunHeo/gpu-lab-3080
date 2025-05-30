FROM python:3.9-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
