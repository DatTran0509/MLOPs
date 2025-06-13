FROM python:3.11.7-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py .

COPY templates/ ./templates/


COPY models/ ./models/

RUN mkdir -p /app/logs

RUN ls -la /app/models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

