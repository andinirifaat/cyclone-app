FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# system deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install torch CPU (dipisah)
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# install sisanya
RUN pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "cyclone_detection_app.py", "--server.port=8501", "--server.address=0.0.0.0"]