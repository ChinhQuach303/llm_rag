import requests
import os

# Base URL của Backend API (khi chạy Docker sẽ map cổng 8000 ra máy host)
API_URL = "http://localhost:8000/v1"

def ingest_data(filepath: str, tenant_id: str = "tenant_test_1"):
    print(f"Bắt đầu Ingest file: {filepath} cho Tenant: {tenant_id}")
    
    with open(filepath, 'rb') as f:
        files = {'file': (os.path.basename(filepath), f, 'application/pdf')}
        data = {'tenant_id': tenant_id}
        
        try:
            response = requests.post(f"{API_URL}/ingest", files=files, data=data)
            print("Status Code:", response.status_code)
            print("Response:", response.json())
        except requests.exceptions.ConnectionError:
            print("Lỗi: Không thể kết nối tới Backend. Hãy chắc chắn Backend đang chạy ở port 8000.")

def test_query(query: str, tenant_id: str = "tenant_test_1"):
    print(f"\n--- Đang Test RAG Query ---")
    data = {
        "query": query,
        "tenant_id": tenant_id,
        "history": []
    }
    
    try:
         # Vì API trả về Streaming Response, ta cần cấu hình stream=True
        response = requests.post(f"{API_URL}/query", json=data, stream=True)
        print("Trạng thái:", response.status_code)
        
        print("Câu trả lời từ RAG: ", end="", flush=True)
        for chunk in response.iter_content(chunk_size=1024):
             if chunk:
                 # decode utf-8 ra chữ
                 print(chunk.decode('utf-8'), end="", flush=True)
                 
    except requests.exceptions.ConnectionError:
            print("Lỗi: Không thể kết nối tới Backend.")

if __name__ == "__main__":
    # 1. Để chạy script này, bạn cần có một file test.pdf cùng thư mục
    # test_file_path = "sample.pdf"
    # if os.path.exists(test_file_path):
    #     ingest_data(test_file_path)
    # else:
    #     print(f"Không tìm thấy file {test_file_path} để test Ingest.")
        
    # 2. Test luồng hỏi đáp
    print("Vui lòng bỏ comment hàm test_query để chạy test.")
    # test_query("Cho tôi tóm tắt về nội dung file vừa tải lên?")
