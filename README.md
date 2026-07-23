# Japanese OCR Analyzer

Ứng dụng Streamlit xử lý ảnh, OCR văn bản tiếng Nhật bằng Gemini 3.5 Flash,
phân tích ngôn ngữ học và xuất báo cáo Word.

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

## Luồng nhiều ảnh

- Chọn nhiều ảnh trong một lần hoặc bổ sung ảnh sau đó.
- OCR từng ảnh hoặc dùng nút OCR toàn bộ.
- Chỉnh sửa, xóa và sắp xếp thứ tự từng ảnh.
- Gộp văn bản theo thứ tự ảnh để phân tích chung và xuất Word.

## Ước tính chi phí

Ứng dụng hiển thị token và chi phí ước tính cho từng lần OCR, lần phân tích
văn bản và tổng phiên. Giá mặc định theo Gemini 3.5 Flash Standard Paid Tier:
input `$0.30/M token`, output gồm thinking token `$2.50/M token`. Có thể chọn
Free Tier để hiển thị chi phí thực tế `$0` cùng giá trị tương đương Paid Tier.

## Kiểm thử

Chạy toàn bộ unit test và integration test mô phỏng Gemini:

```bash
python -m pytest -v
```

Các lệnh gọi Gemini thật cần API key hợp lệ và có thể phát sinh chi phí.
