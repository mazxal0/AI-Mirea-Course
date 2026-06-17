FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

RUN mkdir -p /app/logs
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]