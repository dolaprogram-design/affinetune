FROM python:3.9-slim

RUN apt-get update && apt-get install -y exiftool libgl1 libglib2.0-0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
