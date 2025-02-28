# Use a slim Python base image to reduce size
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies first (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files for the application
# This keeps layers smaller and improves caching
COPY ./app.py .
COPY ./pipelines ./pipelines
COPY ./data ./data
COPY ./churn_model.joblib .
COPY ./processed_data.joblib .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:./mlruns

# Expose the port for the FastAPI application
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
