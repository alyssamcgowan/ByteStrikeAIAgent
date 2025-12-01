FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY slack_handler.py .
COPY agent.py .
COPY rag_pipeline.py .

EXPOSE 8080 

CMD ["python", "slack_handler.py"]