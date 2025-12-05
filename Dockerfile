# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirement files
COPY requirements.txt .
COPY requirements-app.txt .

# Install all packages
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

# Copy the project files
COPY . .

# Ensure data directory exists
RUN mkdir -p data models

# Run the training pipeline
RUN python src/generate_data.py && \
    python src/preprocess.py && \
    python src/train.py && \
    python src/evaluate.py && \
    echo "✓ Model training completed successfully"

# Expose port for Azure App Service
EXPOSE 8000

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
