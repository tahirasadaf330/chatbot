# Dockerfile
FROM python:3.11-slim

# System deps for faiss & PDFs
RUN apt-get update && apt-get install -y build-essential libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index at image build time if you COPY pdfs/ in; otherwise do it at runtime.
# RUN python build_index.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
