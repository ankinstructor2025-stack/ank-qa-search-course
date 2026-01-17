FROM python:3.11-slim
WORKDIR /app
# 依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# アプリとデータをまとめてコピー
COPY . /app
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
