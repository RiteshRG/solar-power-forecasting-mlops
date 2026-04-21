# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make start script executable
RUN chmod +x start.sh

# Expose both ports
EXPOSE 8501
EXPOSE 8000

# Run both Streamlit and FastAPI
CMD ["./start.sh"]