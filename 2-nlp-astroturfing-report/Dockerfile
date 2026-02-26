# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Ensure the credentials file path from .env exists or handle it via volume/env
# For Cloud Run, it's better to use Service Account ADC or mount secrets, 
# but for this specific request, we'll assume the key is provided.

# Set the entry point to run the scraper
# Usage: docker run ... [url]
ENTRYPOINT ["python", "-m", "src.forensics.manual_scrapping"]
