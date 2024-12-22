# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install FastAPI, Uvicorn, and joblib (and other dependencies you may need)
RUN pip install --no-cache-dir fastapi uvicorn joblib

# Expose port 8000 to access the app
EXPOSE 8000

# Run the application with Uvicorn as the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
