FROM python:3.9-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements 먼저 복사 → 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# ✅ ONNX 파일이 없으면 export 스크립트 실행
# RUN python3 -c "import pathlib; not pathlib.Path('resnet50.onnx').exists() and __import__('export_onnx')"
# CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT ["sh", "-c", "python3 -c 'import export_onnx; export_onnx.export_resnet()' && uvicorn api_server:app --host 0.0.0.0 --port 8000"]
