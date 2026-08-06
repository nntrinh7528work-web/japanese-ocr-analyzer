---
title: Japanese OCR Analyzer
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.53.1
app_file: app.py
pinned: false
---

# Japanese OCR Analyzer

Ứng dụng Streamlit xử lý ảnh/PDF, OCR văn bản tiếng Nhật hoặc tiếng Anh bằng
Gemini và phân tích từ vựng, kanji, ngữ pháp bằng tiếng Việt. Ứng dụng dùng SDK
`google-genai` hiện hành và cho phép chọn model OCR/phân tích trên giao diện.

## Chạy ứng dụng

1. Cài dependency: `python -m pip install -r requirements.txt`
2. Đặt `GEMINI_API_KEY` trong `.env`
3. Chạy: `streamlit run app.py`

## Triển khai lâu dài

Ứng dụng tương thích với Streamlit Community Cloud. Push repository lên GitHub,
chọn `app.py` làm entrypoint và thêm secret sau trong Advanced settings:

```toml
GEMINI_API_KEY = "your_key_here"
```

Không commit file `.env` hoặc `.streamlit/secrets.toml`.

## Đồng bộ Notion cá nhân

Tạo một Notion internal connection có quyền đọc, chèn và cập nhật nội dung;
chia sẻ một trang cha trống với connection rồi thêm hai secret sau:

```toml
NOTION_TOKEN = "secret_..."
NOTION_PARENT_PAGE_ID = "..."
PUBLIC_APP_URL = "https://japanese-ocr-analyzer-vn.streamlit.app/"
```

Ở lần đồng bộ đầu tiên, app tự tạo hai bảng liên kết `Bài phân tích` và
`Mục cần học`, cùng các view ôn tập. Token chỉ được đọc từ environment hoặc
Streamlit Secrets; SQLite và file JSON không lưu token. Có thể điền thêm các
database/data-source ID trong `.streamlit/secrets.toml.example` để giữ cấu hình
ổn định khi máy chủ được tạo lại.

Sau khi phân tích hoàn tất, app lưu kết quả trước rồi đồng bộ Notion trong nền.
Lỗi quyền hoặc mạng không làm mất kết quả Gemini; giao diện có trạng thái, link
mở trang Notion và nút thử lại.

Mỗi kết quả đồng bộ được lưu thành một phiên bản độc lập theo `OCR hash` và
`Analysis hash`; chạy lại cùng OCR với kết quả mới không ghi đè bài cũ. Trang
`Bài phân tích` chứa nội dung đầy đủ và file JSON gốc có SHA-256 để kiểm tra tính
toàn vẹn. Các cột OCR, hướng dẫn dịch, bản dịch, từ vựng, kanji/cụm từ, từ nối,
ngữ pháp, mẫu câu, câu dài, token và chi phí là bản tóm lược để tìm kiếm.

Bảng `Mục cần học` nhận toàn bộ từ vựng cùng kanji, cụm từ, từ nối, ngữ pháp,
mẫu câu và câu dài. App không ghi đè các cột học tập do người dùng quản lý như
`Trạng thái`, `Ngày ôn tiếp`, `Lần ôn gần nhất` và `Số lần ôn` khi đồng bộ lại.

## Luồng nhiều ảnh

- Chọn nhiều ảnh trong một lần hoặc bổ sung ảnh sau đó.
- OCR từng ảnh hoặc dùng nút OCR toàn bộ.
- Chỉnh sửa, xóa và sắp xếp thứ tự từng ảnh.
- Gộp văn bản theo thứ tự ảnh để phân tích chung và xuất Word.
- Tác vụ phân tích lưu kết quả từng trang để có thể tiếp tục sau lỗi tạm thời.
- Kết quả được sắp theo thứ tự trang và đánh lại STT tự động; mục từ nối phân
  biệt liên từ, trợ từ nối, cụm diễn ngôn, quan hệ logic và sắc thái sử dụng.

## Lịch sử và dữ liệu

- Ảnh, OCR, kết quả phân tích, ngân sách và lịch sử hội thoại được cách ly theo
  từng phiên truy cập.
- Dữ liệu cục bộ được dọn sau 30 ngày. SQLite trên Streamlit Community Cloud
  không phải kho lưu trữ vĩnh viễn và có thể bị xóa khi ứng dụng được triển khai
  lại hoặc máy chủ được tái tạo. Muốn đồng bộ lâu dài giữa nhiều thiết bị cần
  kết nối một cơ sở dữ liệu ngoài như PostgreSQL/Supabase.

## Ước tính chi phí

Ứng dụng hiển thị token và chi phí ước tính cho từng lần OCR, lần phân tích
văn bản và tổng phiên bằng USD hoặc JPY. Giá Gemini 3.5 Flash Standard Paid
Tier đang cấu hình là input `$1.50/M token`, output gồm thinking token
`$9.00/M token`. Có thể chọn Free Tier để hiển thị chi phí thực tế `$0` cùng
giá trị tương đương Paid Tier. Đây là ước tính theo token, không phải số dư lấy
trực tiếp từ Google Cloud Billing.

## Kiểm thử

Chạy toàn bộ unit test và integration test mô phỏng Gemini:

```bash
python -m pytest -v
```

Các lệnh gọi Gemini thật cần API key hợp lệ và có thể phát sinh chi phí.
