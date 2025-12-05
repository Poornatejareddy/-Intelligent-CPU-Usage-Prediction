# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files into the container at /app
COPY requirements.txt requirements-app.txt ./

# Install all needed packages (both training and app dependencies)
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run the training pipeline to generate the model
# First generate the data, then preprocess, train, and evaluate
RUN python src/generate_data.py && \
    python src/preprocess.py && \
    python src/train.py && \
    python src/evaluate.py && \
    echo "✓ Model training completed successfully"

# Expose port 8000 for Azure App Service
EXPOSE 8000

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
