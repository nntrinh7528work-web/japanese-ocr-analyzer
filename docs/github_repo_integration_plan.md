# Kế hoạch Tích hợp GitHub Repos cho Japanese NLP

## 1. Khảo sát Hiện trạng Repo
Dựa vào quá trình khảo sát mã nguồn hiện tại, tôi đã xác định được:
- **Nơi text sau OCR được đưa vào pipeline phân tích:** Tại file `app.py` (khoảng dòng 580-630), text từ `st.session_state.ocr_items` được gom lại thông qua hàm `analysis_pages(items)`, sau đó đưa vào `text_analyzer.run_page_analyses` hoặc background job (`worker.py` gọi `run_page_analyses_pipeline` / `run_verified_analysis`).
- **Nơi xử lý tiếng Nhật hiện tại:** Logic phân tích AI nằm ở `modules/text_analyzer.py`, `modules/deepseek_analyzer.py`, `modules/gemini_reviewer.py`, và được điều phối bởi `modules/analysis_pipeline.py`.
- **Nơi quản lý dependency:** File `requirements.txt` ở thư mục gốc.
- **Nơi phù hợp nhất để thêm NLP helper layer:** Tạo một layer phụ trợ độc lập (ví dụ `modules/japanese_nlp_helper.py` và `modules/post_ocr_japanese_check.py`), sau đó chèn lệnh gọi nhẹ vào `app.py` ngay sau khi có text OCR và trước khi gửi đi phân tích LLM nặng. Cách này sẽ không phá vỡ pipeline cũ.

## 2. Đánh giá Tích hợp các Repo/Thư viện

Dưới đây là bảng đánh giá 3 repo được yêu cầu:

| Repo | Loại | Có thể cài trực tiếp không | Tác dụng với app | Nên tích hợp ngay hay không |
|---|---|---|---|---|
| [taishi-i/nagisa](https://github.com/taishi-i/nagisa) | Thư viện Runtime (Python) | Cài được trực tiếp qua `pip install nagisa` | Phân tách từ (tokenization) và gán nhãn từ loại (POS tagging) tiếng Nhật dựa trên RNN. Rất nhẹ và dễ sử dụng. | **Nên tích hợp ngay** (ở Bước 2) để tạo lớp xử lý NLP phụ trợ. |
| [msr2903/himotoki](https://github.com/msr2903/himotoki) | Thư viện Runtime (Python) | Có thể cài qua pip hoặc clone từ GitHub | Hỗ trợ Tokenizer, morphological analyzer, romanization và tra cứu JMDict. | **Khoan tích hợp ngay** (Đánh giá ở Bước 5). Cần kiểm tra độ ổn định và có trùng lặp tính năng với nagisa hay không. |
| [taishi-i/awesome-japanese-nlp-resources](https://github.com/taishi-i/awesome-japanese-nlp-resources) | Danh sách tài nguyên (Awesome List) | Không (chỉ là tài liệu Markdown) | Cung cấp cái nhìn tổng quan về các công cụ, bộ dataset và thư viện tiếng Nhật hiện có trên thị trường. | **Không tích hợp vào code**. Chỉ dùng làm tài liệu tham khảo (Bước 6). |
