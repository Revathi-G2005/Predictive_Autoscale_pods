# ml_predictor/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy predictor code
COPY . .

CMD ["python", "predictor.py"]
