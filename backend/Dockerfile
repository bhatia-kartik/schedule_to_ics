FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tesseract-ocr libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
