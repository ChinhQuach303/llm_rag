import pdfplumber
import os

# Đường dẫn thư mục chứa file PDF
pdf_folder = r"C:\Users\ADMIN\code\ASM2\pdf"

# Tạo thư mục hoặc file để lưu văn bản (tùy chọn)
output_file = "output_text.txt"

# Mở file để lưu văn bản
with open(output_file, "w", encoding="utf-8") as f:
    # Lặp qua tất cả file trong thư mục
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):  # Kiểm tra file có đuôi .pdf
            file_path = os.path.join(pdf_folder, filename)
            print(f"Đang đọc file: {filename}")
            f.write(f"\n=== File: {filename} ===\n")
            
            # Mở và đọc file PDF
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()  # Trích xuất văn bản
                    if text:
                        f.write(f"Trang {page.page_number}:\n")
                        f.write(text + "\n")
                        f.write("-" * 50 + "\n")