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

# Ensure the app can find the src module
ENV PYTHONPATH=/app

# Default command: run the integrated scraping flow
# Usage: docker run --env-file .env reddit-forensics [URL]
ENTRYPOINT ["python", "-m", "src.forensics.manual_scrapping"]
CMD ["--help"]
