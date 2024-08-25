# Use a multi-stage build to reduce the final image size
# Stage 1: Build stage
FROM --platform=$BUILDPLATFORM python:3.11-slim as builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh

# Copy the requirements.txt file
COPY requirements.txt ./

# Install Python dependencies using uv
RUN /root/.cargo/bin/uv pip install --system --no-cache -r requirements.txt

# Stage 2: Final stage
FROM --platform=$TARGETPLATFORM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the installed dependencies from the builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /root/.cargo /root/.cargo

# Copy the rest of the application code
COPY app.py .
COPY pravah/ ./pravah/

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]