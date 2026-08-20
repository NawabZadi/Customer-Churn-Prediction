#Docker creates a container containing everything your application needs.
#The Dockerfile is the recipe for creating that container.

# Use Python as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY app ./app
COPY src ./src
COPY models ./models

# Start the FastAPI application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]