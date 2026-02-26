FROM python:3.10-slim

WORKDIR /app

# Cài đặt dependencies hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Lệnh chạy mặc định được override bởi docker-compose, nhưng giữ lại để tham khảo
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]