import requests
import time
import os

import jwt
from dotenv import load_dotenv

load_dotenv("../.env")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

token = jwt.encode({"tenant_id": "tenant_docker_test"}, SECRET_KEY, algorithm=ALGORITHM)

BASE_URL = "http://localhost:8000/v1"
headers = {"Authorization": f"Bearer {token}"}

def test_ingestion():
    # Write a dummy pdf
    with open("dummy.pdf", "wb") as f:
        f.write(b"%PDF-1.4\n%...\nHello World\n%%EOF")
        
    print("Uploading document...")
    with open("dummy.pdf", "rb") as f:
        files = {"file": ("dummy.pdf", f, "application/pdf")}
        data = {"tenant_id": "tenant_docker_test"}
        
        response = requests.post(
            f"{BASE_URL}/ingest",
            files=files,
            data=data,
            headers=headers
        )
        
    if response.status_code == 200:
        task_id = response.json().get("task_id")
        print(f"Ingestion started. Task ID: {task_id}")
        return task_id
    else:
        print(f"Ingestion failed: {response.text}")
        return None

if __name__ == "__main__":
    test_ingestion()
