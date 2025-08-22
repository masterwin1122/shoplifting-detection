FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    curl \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

ENV FLASK_APP=main.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]

