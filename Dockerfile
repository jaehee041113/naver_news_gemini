FROM python:3.11-slim

# 시스템 패키지 설치 (lxml, bs4 때문에 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# requirements 먼저 복사 (캐싱 최적화)
COPY gemini_req.txt .

# pip 업그레이드 + 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r gemini_req.txt

# 앱 코드 복사
COPY . .

# 포트 오픈 (Gradio)
EXPOSE 7860

# 실행
CMD ["python", "app.py"]