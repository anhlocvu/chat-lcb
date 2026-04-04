chat lcb
an open-source AI assistant
# Chat LCB - Trợ Lý AI Thông Minh (Offline)

Chào mừng bạn đến với **Chat LCB**! Đây là một ứng dụng AI Chatbot được xây dựng trên kiến trúc **RAG (Retrieval-Augmented Generation)**, cho phép AI học và trả lời dựa trên dữ liệu riêng của bạn (như PDF, TXT) một cách hoàn toàn miễn phí và bảo mật trên máy tính cá nhân.

---

## 💻 Cấu hình yêu cầu
Để hệ thống chạy mượt mà, máy tính của bạn nên có:
- **RAM:** 
  - 8GB trở lên (Nếu dùng Gemma 2 2B)
  - 16GB trở lên (Khuyến nghị cho Gemma 2 9B - Thông minh nhất)
- **CPU:** Chip đời mới (i5 Gen 10 hoặc Ryzen 5 trở lên).
- **Ổ cứng:** Còn trống ít nhất 10GB.
- **Hệ điều hành:** Windows 10/11, Linux hoặc macOS.

---

## 🚀 Hướng dẫn cài đặt nhanh

### Bước 1: Cài đặt Ollama (Bộ não AI)
1. Tải Ollama tại: [https://ollama.com/download](https://ollama.com/download)
2. Cài đặt và mở Ollama lên.
3. Mở Terminal (PowerShell) và gõ lệnh sau để tải mô hình AI:
   ```powershell
   ollama run gemma2:9b
   ```
   *(Chờ tải xong 100% rồi nhấn Ctrl + D để thoát).*

### Bước 2: Cài đặt Python và Công cụ quản lý
1. Cài đặt Python bản 3.10 trở lên.
2. Cài đặt công cụ `uv` (giúp chạy Python cực nhanh):
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

### Bước 3: Chuẩn bị dự án
1. Tải (Clone) dự án này về máy.
2. Mở thư mục dự án trong Terminal và chạy lệnh sau để tự động cài đặt mọi thư viện:
   ```powershell
   uv sync
   ```

---

## 🧠 Cách nạp dữ liệu cho AI (Đào tạo)
Bạn có thể "dạy" AI kiến thức riêng của bạn bằng cách:
1. Copy các file tài liệu của bạn (định dạng `.pdf` hoặc `.txt`) vào thư mục `data/`.
2. Chạy file `Train.bat` (nhấn đúp chuột).
   - Hệ thống sẽ xóa "não bộ" cũ và nạp dữ liệu mới nhất từ thư mục `data/` vào AI.
   - Khi hoàn tất, AI sẽ có kiến thức từ các tài liệu này.

---

## 💬 Cách sử dụng Chatbot
1. Chạy file `Run.bat` (nhấn đúp chuột).
2. Khi thấy thông báo `Uvicorn running on http://0.0.0.0:8000`, hãy mở trình duyệt web lên.
3. Nhập địa chỉ: `http://localhost:8000` và bắt đầu trò chuyện với **Chat LCB**.

### Các tính năng đặc biệt:
- **Trí nhớ dài hạn:** AI nhớ toàn bộ lịch sử trò chuyện của bạn trong cùng một phiên.
- **Đa phiên (Sessions):** Bạn có thể tạo nhiều cuộc hội thoại khác nhau bằng nút "Phiên mới".
- **Hỗ trợ trợ năng:** Giao diện được tối ưu cho trình đọc màn hình (như NVDA) với các Heading (H3) rõ ràng cho từng tin nhắn.




---
*Phát triển bởi Vũ Anh Lộc (Lc_Boy).*
