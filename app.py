"""Streamlit entry point for Japanese/English OCR Analyzer."""

from __future__ import annotations

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from config import (
    GEMINI_MODEL_TEXT,
    GEMINI_MODEL_VISION,
    MAX_IMAGE_SIZE_MB,
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_MB,
    SUPPORTED_UPLOAD_FORMATS,
)
from modules.cost_estimator import estimate_cost, format_cost, sum_costs
from modules.doc_exporter import export_to_docx
from modules.multi_image_workflow import add_upload_items, combined_text, move_image_item
from modules.ocr_engine import run_ocr
from modules.result_exporter import analysis_json_bytes, default_export_stem, markdown_bytes, safe_export_stem
from modules import session_store
import modules.text_analyzer as text_analyzer
from modules.web_scraper import fetch_article


text_analyzer = importlib.reload(text_analyzer)


st.set_page_config(page_title="Japanese / English OCR Analyzer", page_icon="🔍", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzone"] {
        min-height: 135px; border: 2px dashed #4f8bf9; border-radius: 12px;
    }
    [data-testid="stFileUploaderDropzone"] button { min-height: 44px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in {
    "image_items": [],
    "analysis": None,
    "partial_page_analyses": [],
    "upload_messages": [],
    "upload_errors": [],
    "uploader_version": 0,
    "camera_version": 0,
    "session_id": None,
    "session_restored": False,
    "url_scrape_result": None,
    "url_analysis_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Session persistence: restore or create ──────────────────────────────
session_store.cleanup_old_sessions(max_age_hours=24)

query_sid = st.query_params.get("session", "").strip()

if not st.session_state.session_restored:
    if query_sid and session_store.session_exists(query_sid):
        # Restore previous session from SQLite.
        st.session_state.session_id = query_sid
        saved_items = session_store.load_image_items(query_sid)
        if saved_items:
            st.session_state.image_items = saved_items
        saved_analysis, saved_partial = session_store.load_analysis(query_sid)
        if saved_analysis:
            st.session_state.analysis = saved_analysis
        if saved_partial:
            st.session_state.partial_page_analyses = saved_partial
        session_store.update_session_timestamp(query_sid)
    else:
        # Create a brand-new session.
        new_sid = session_store.generate_session_id()
        session_store.create_session(new_sid)
        st.session_state.session_id = new_sid
        st.query_params["session"] = new_sid
    st.session_state.session_restored = True

# Ensure query param stays in sync.
if st.session_state.session_id and st.query_params.get("session") != st.session_state.session_id:
    st.query_params["session"] = st.session_state.session_id


def _persist_items() -> None:
    """Save current image items to SQLite."""
    sid = st.session_state.session_id
    if sid:
        session_store.save_image_items(sid, st.session_state.image_items)
        session_store.update_session_timestamp(sid)


def _persist_analysis() -> None:
    """Save current analysis and partial results to SQLite."""
    sid = st.session_state.session_id
    if sid:
        session_store.save_analysis(
            sid,
            st.session_state.analysis,
            st.session_state.partial_page_analyses,
        )
        session_store.update_session_timestamp(sid)


COLUMN_LABELS = {
    "english": {
        "num": "STT",
        "word": "Từ",
        "base_form": "Dạng gốc",
        "part_of_speech": "Loại từ",
        "meaning": "Nghĩa tiếng Việt",
        "cefr": "Cấp độ CEFR",
        "example": "Ví dụ trong bài",
        "difficulty": "Độ khó",
        "phrase": "Từ/Cụm từ",
        "type": "Loại",
        "note": "Ghi chú",
        "function": "Chức năng",
        "register": "Sắc thái",
    },
    "japanese": {
        "num": "STT",
        "word": "Từ gốc",
        "reading": "Phiên âm",
        "type": "Loại từ",
        "meaning": "Nghĩa",
        "jlpt": "JLPT",
        "example": "Ví dụ",
        "difficulty": "Độ khó",
        "kanji": "Kanji",
        "onyomi": "Âm On",
        "kunyomi": "Âm Kun",
        "vocab": "Từ vựng trong bài",
        "role": "Vai trò",
        "phrase": "Từ/Cụm",
    },
}


def display_rows(rows: list[dict], language: str) -> list[dict]:
    labels = COLUMN_LABELS.get(language, {})
    return [{labels.get(key, key): value for key, value in row.items()} for row in rows]


def render_example(label: str, text: str | None, hiragana: str | None = None) -> None:
    if not text:
        return
    st.markdown(label)
    st.info(text)
    if hiragana:
        st.caption(f"ひらがな: {hiragana}")


def render_important_vocabulary(items: list[dict]) -> None:
    if not items:
        st.info("Chưa trích xuất được từ vựng quan trọng.")
        return

    st.subheader("⭐ Từ vựng khó — Giải thích chi tiết")
    for item in items:
        level = item.get("jlpt") or item.get("cefr") or item.get("difficulty") or ""
        label = item.get("word", "")
        if item.get("reading"):
            label = f"{label}・{item['reading']}"
        page_tag = f" — {item['page_label']}" if item.get("page_label") else ""
        with st.expander(f"📖 {label}   {level}{page_tag}".strip()):
            col1, col2 = st.columns(2)
            with col1:
                if item.get("type") or item.get("part_of_speech"):
                    st.markdown(f"**Loại từ:** {item.get('type') or item.get('part_of_speech')}")
                if item.get("meaning"):
                    st.markdown(f"**Ý nghĩa:** {item['meaning']}")
                if item.get("vn_meaning"):
                    st.markdown(f"**Nghĩa tiếng Việt:** {item['vn_meaning']}")
                if item.get("definition"):
                    st.markdown(f"**Definition:** {item['definition']}")
                if item.get("base_form"):
                    st.markdown(f"**Dạng gốc:** {item['base_form']}")
                if item.get("related") and item["related"] not in ("Không có", "None", "N/A"):
                    st.markdown(f"**Từ liên quan / Related:** {item['related']}")
            with col2:
                example_text = item.get("example_text") or item.get("example")
                render_example("**📌 Ví dụ trong bài:**", example_text, item.get("example_text_hiragana"))
                render_example("**✏️ Ví dụ 1:**", item.get("example_1"), item.get("example_1_hiragana"))
                render_example("**✏️ Ví dụ 2:**", item.get("example_2"), item.get("example_2_hiragana"))
            if item.get("note") and item["note"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Lưu ý:** {item['note']}")
            if item.get("mistake") and item["mistake"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Common mistake:** {item['mistake']}")


def render_grammar_points(items: list[dict]) -> None:
    if not items:
        st.info("Chưa trích xuất được mục ngữ pháp riêng. Xem nội dung gốc bên dưới.")
        return

    for point in items:
        level = point.get("level") or ""
        page_tag = f" — {point['page_label']}" if point.get("page_label") else ""
        with st.expander(f"📌 {point.get('name', '')}   {level}{page_tag}".strip()):
            col1, col2 = st.columns(2)
            with col1:
                if point.get("structure"):
                    st.markdown(f"**Công thức / Structure:** `{point['structure']}`")
                if point.get("rule"):
                    st.markdown(f"**Quy tắc:** {point['rule']}")
                if point.get("meaning"):
                    st.markdown(f"**Ý nghĩa:** {point['meaning']}")
                if point.get("usage"):
                    st.markdown(f"**Cách dùng:** {point['usage']}")
                if point.get("explanation"):
                    st.markdown(f"**Giải thích:** {point['explanation']}")
            with col2:
                if point.get("example"):
                    st.markdown("**📌 Ví dụ trong bài:**")
                    st.code(point["example"], language=None)
                if point.get("example_analysis"):
                    st.markdown(f"**Phân tích ví dụ:** {point['example_analysis']}")
                if point.get("example_1"):
                    st.markdown(f"**✏️ Ví dụ 1:** {point['example_1']}")
                if point.get("example_2"):
                    st.markdown(f"**✏️ Ví dụ 2:** {point['example_2']}")
            if point.get("note") and point["note"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Lưu ý:** {point['note']}")
            if point.get("mistake") and point["mistake"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Common mistake:** {point['mistake']}")


def clear_analysis() -> None:
    st.session_state.analysis = None
    st.session_state.partial_page_analyses = []
    _persist_analysis()


def analysis_pages(items: list[dict]) -> list[dict]:
    pages = []
    for index, item in enumerate(items, 1):
        text = str(item.get("edited_text") or "").strip()
        if not text:
            continue
        result = item.get("ocr_result") or {}
        pages.append(
            {
                "page_index": index,
                "page_name": item.get("name") or f"Trang {index}",
                "text": text,
                "notes": result.get("ocr_notes", []),
            }
        )
    return pages


def add_sources(sources: list[tuple[str, bytes]]) -> bool:
    items, added, errors = add_upload_items(st.session_state.image_items, sources)
    st.session_state.image_items = items
    st.session_state.upload_messages = []
    st.session_state.upload_errors = errors
    if added:
        clear_analysis()
        st.session_state.upload_messages.append(f"Đã thêm {len(added)} ảnh/trang PDF.")
        _persist_items()
    return bool(added)


def remove_image(item_id: str) -> None:
    st.session_state.image_items = [item for item in st.session_state.image_items if item["id"] != item_id]
    clear_analysis()
    _persist_items()


def run_item_ocr(item: dict) -> None:
    item["ocr_error"] = None
    try:
        result = run_ocr(item["processed_image_bytes"], item["report"])
        item["ocr_result"] = result
        item["edited_text"] = result["clean_text"]
    except Exception as exc:
        item["ocr_error"] = str(exc)


def current_budget_status(
    budget_jpy: float,
    spent_before_jpy: float,
    session_cost_usd: float,
    usd_to_jpy: float,
) -> dict[str, float]:
    """Estimate remaining API budget in JPY for the current app session."""
    budget_value = max(0.0, float(budget_jpy or 0))
    spent_before = max(0.0, float(spent_before_jpy or 0))
    session_spent = max(0.0, float(session_cost_usd or 0) * float(usd_to_jpy or 0))
    total_spent = spent_before + session_spent
    remaining = max(0.0, budget_value - total_spent)
    return {
        "budget_jpy": budget_value,
        "spent_before_jpy": spent_before,
        "session_spent_jpy": session_spent,
        "total_spent_jpy": total_spent,
        "remaining_jpy": remaining,
        "used_ratio": total_spent / budget_value if budget_value else 0.0,
        "remaining_ratio": remaining / budget_value if budget_value else 0.0,
    }


st.title("🔍 Japanese / English OCR Analyzer")
st.caption("Tải nhiều ảnh hoặc PDF, OCR từng trang hoặc toàn bộ, rồi chọn phân tích tiếng Nhật hoặc tiếng Anh.")

items = st.session_state.image_items
st.sidebar.header("📚 Bộ ảnh phân tích")
st.sidebar.metric("Số ảnh", len(items))
st.sidebar.metric("Đã OCR", sum(bool(item["ocr_result"]) for item in items))
if items and st.sidebar.button("🗑️ Xóa toàn bộ bộ ảnh", width="stretch"):
    st.session_state.image_items = []
    st.session_state.analysis = None
    st.session_state.uploader_version += 1
    st.session_state.camera_version += 1
    _persist_items()
    _persist_analysis()
    st.rerun()

# ── Session info in sidebar ──
if st.session_state.session_id:
    st.sidebar.divider()
    st.sidebar.subheader("💾 Phiên làm việc")
    st.sidebar.code(st.session_state.session_id, language=None)
    st.sidebar.caption(
        "Mã phiên tự động. Khi thoát app, mở lại link có mã này để khôi phục toàn bộ tiến trình. "
        "Phiên được giữ tối đa 24 giờ."
    )
st.sidebar.divider()
st.sidebar.header("⚙️ Settings")
show_preprocessing = st.sidebar.toggle("Hiển thị chi tiết tiền xử lý", value=True)
st.sidebar.caption(f"Ảnh tối đa {MAX_IMAGE_SIZE_MB} MB · PDF tối đa {MAX_PDF_SIZE_MB} MB/{MAX_PDF_PAGES} trang")
analysis_language = st.sidebar.radio(
    "Ngôn ngữ phân tích",
    options=["japanese", "english"],
    format_func=lambda value: "Tiếng Nhật (Kanji/JLPT)" if value == "japanese" else "Tiếng Anh (CEFR/Grammar)",
    horizontal=False,
)
if st.session_state.get("_last_analysis_language") not in (None, analysis_language):
    clear_analysis()
st.session_state["_last_analysis_language"] = analysis_language
st.sidebar.subheader("💰 Ước tính chi phí")
billing_tier = st.sidebar.radio(
    "Gói Gemini",
    options=["free", "paid"],
    format_func=lambda value: "Free Tier" if value == "free" else "Paid Tier (Standard)",
    horizontal=True,
)
usd_to_jpy = st.sidebar.number_input("Tỷ giá USD → JPY", min_value=1.0, value=155.0, step=1.0)
st.sidebar.caption("Gemini 2.5 Flash: input $0.30/M token, output $2.50/M token.")
st.sidebar.markdown("[Xem bảng giá Gemini chính thức](https://ai.google.dev/gemini-api/docs/pricing)")
st.sidebar.subheader("🏦 Theo dõi ngân sách API")
api_budget_jpy = st.sidebar.number_input(
    "Số tiền đã nạp/ngân sách (JPY)",
    min_value=0.0,
    value=0.0,
    step=1_000.0,
    help="Nhập số tiền bạn đã nạp hoặc muốn dùng làm ngân sách theo dõi.",
)
api_spent_before_jpy = st.sidebar.number_input(
    "Đã dùng trước đó (JPY)",
    min_value=0.0,
    value=0.0,
    step=100.0,
    help="Nhập thủ công số tiền đã dùng trước phiên hiện tại nếu bạn muốn theo dõi nhiều lần dùng.",
)
st.sidebar.caption("Số dư này là ước tính trong app, không phải số dư chính thức từ Google Billing.")
with st.sidebar.expander("💡 Luồng sử dụng"):
    st.write("1. Thêm ảnh hoặc PDF từ máy/ứng dụng Files.")
    st.write("2. Mỗi trang PDF sẽ được chuyển thành một ảnh.")
    st.write("3. OCR từng trang hoặc OCR toàn bộ.")
    st.write("4. Phân tích chung tất cả nội dung đã OCR.")

tab_ocr, tab_url = st.tabs(["📷 Phân tích từ Ảnh / PDF", "🌐 Phân tích từ URL"])

with tab_ocr:
    with st.expander("➕ Thêm ảnh hoặc PDF vào bộ phân tích", expanded=not items):
        upload_tab, camera_tab = st.tabs(["📁 Chọn ảnh/PDF từ máy", "📷 Chụp thêm ảnh"])
        with upload_tab:
            uploaded_files = st.file_uploader(
                "Kéo thả hoặc bấm nút để chọn ảnh/PDF",
                type=SUPPORTED_UPLOAD_FORMATS,
                accept_multiple_files=True,
                key=f"multi_uploader_{st.session_state.uploader_version}",
                help=f"PDF tối đa {MAX_PDF_SIZE_MB} MB và {MAX_PDF_PAGES} trang. Trên iPhone, chọn Browse để mở Files.",
            )
            st.caption(f"Hỗ trợ PDF tối đa {MAX_PDF_SIZE_MB} MB. Trên iPhone, chọn Browse/Duyệt để mở ứng dụng Files.")
            if uploaded_files and st.button("➕ Thêm file đã chọn", type="primary", width="stretch"):
                with st.spinner("Đang xử lý file upload..."):
                    added_any = add_sources([(file.name, file.getvalue()) for file in uploaded_files])
                if added_any:
                    st.session_state.uploader_version += 1
                    st.rerun()
            for message in st.session_state.upload_messages:
                st.success(f"✅ {message}")
            for error in st.session_state.upload_errors:
                st.error(f"❌ Không thể thêm file: {error}")
        with camera_tab:
            camera_file = st.camera_input(
                "Chụp ảnh văn bản",
                key=f"camera_{st.session_state.camera_version}",
            )
            if camera_file and st.button("➕ Thêm ảnh vừa chụp", width="stretch"):
                if add_sources([(f"camera_{len(items) + 1}.jpg", camera_file.getvalue())]):
                    st.session_state.camera_version += 1
                    st.rerun()

    if not items:
        st.info("Hãy thêm một hoặc nhiều ảnh/PDF để bắt đầu.")
    else:
        st.subheader(f"Ảnh/trang PDF trong bộ phân tích ({len(items)})")
        controls_left, controls_middle, controls_right = st.columns(3)
        with controls_left:
            if st.button("🔍 OCR tất cả ảnh chưa xử lý", type="primary", width="stretch"):
                pending = [item for item in items if not item["ocr_result"]]
                if not pending:
                    st.info("Tất cả ảnh/trang đã có OCR.")
                else:
                    progress = st.progress(0, text="Đang OCR...")
                    lock = threading.Lock()
                    done_count = 0

                    def _ocr_work(item_to_ocr: dict) -> dict:
                        item_to_ocr["ocr_error"] = None
                        try:
                            result = run_ocr(item_to_ocr["processed_image_bytes"], item_to_ocr["report"])
                            item_to_ocr["ocr_result"] = result
                            item_to_ocr["edited_text"] = result["clean_text"]
                        except Exception as exc:
                            item_to_ocr["ocr_error"] = str(exc)
                        return item_to_ocr

                    with ThreadPoolExecutor(max_workers=min(3, len(pending))) as pool:
                        futures = {pool.submit(_ocr_work, p): p for p in pending}
                        for future in as_completed(futures):
                            future.result()
                            done_count += 1
                            progress.progress(done_count / len(pending), text=f"Đã OCR {done_count}/{len(pending)}")
                    st.session_state.image_items = list(items)
                    _persist_items()
                    progress.empty()
                    clear_analysis()
                    st.rerun()
        with controls_middle:
            if st.button("🔁 OCR/OCR lại toàn bộ ảnh", width="stretch"):
                progress = st.progress(0, text="Đang OCR toàn bộ...")
                done_count = 0

                def _ocr_work_all(item_to_ocr: dict) -> dict:
                    item_to_ocr["ocr_error"] = None
                    try:
                        result = run_ocr(item_to_ocr["processed_image_bytes"], item_to_ocr["report"])
                        item_to_ocr["ocr_result"] = result
                        item_to_ocr["edited_text"] = result["clean_text"]
                    except Exception as exc:
                        item_to_ocr["ocr_error"] = str(exc)
                    return item_to_ocr

                with ThreadPoolExecutor(max_workers=min(3, len(items))) as pool:
                    futures = {pool.submit(_ocr_work_all, it): it for it in items}
                    for future in as_completed(futures):
                        future.result()
                        done_count += 1
                        progress.progress(done_count / len(items), text=f"Đã OCR {done_count}/{len(items)}")
                st.session_state.image_items = list(items)
                _persist_items()
                progress.empty()
                clear_analysis()
                st.rerun()
        with controls_right:
            ready_count = sum(bool(item.get("edited_text", "").strip()) for item in items)
            st.info(f"Sẵn sàng phân tích: {ready_count}/{len(items)} ảnh")

        for index, item in enumerate(items, 1):
            status = "✅ Đã OCR" if item["ocr_result"] else "⏳ Chưa OCR"
            with st.expander(f"Ảnh {index}: {item['name']} · {status}", expanded=len(items) == 1):
                original_col, processed_col = st.columns(2)
                with original_col:
                    st.caption("Ảnh gốc")
                    st.image(item["original_image_bytes"], width="stretch")
                with processed_col:
                    st.caption("Ảnh đã xử lý")
                    st.image(item["processed_image_bytes"], width="stretch")

                report = item["report"]
                if show_preprocessing:
                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Chất lượng", report["quality_level"])
                    metric2.metric("Góc xoay", f"{report['rotation_detected']}°")
                    metric3.metric("Blur score", f"{report['blur_score']:.1f}")

                action1, action2, action3, action4 = st.columns(4)
                if action1.button("🔍 OCR ảnh này", key=f"ocr_{item['id']}", width="stretch"):
                    with st.spinner(f"Đang OCR {item['name']}..."):
                        run_item_ocr(item)
                    _persist_items()
                    clear_analysis()
                    st.rerun()
                if action2.button("⬆️ Lên", key=f"up_{item['id']}", disabled=index == 1, width="stretch"):
                    st.session_state.image_items = move_image_item(items, item["id"], -1)
                    clear_analysis()
                    st.rerun()
                if action3.button("⬇️ Xuống", key=f"down_{item['id']}", disabled=index == len(items), width="stretch"):
                    st.session_state.image_items = move_image_item(items, item["id"], 1)
                    clear_analysis()
                    st.rerun()
                if action4.button("🗑️ Xóa", key=f"remove_{item['id']}", width="stretch"):
                    remove_image(item["id"])
                    st.rerun()

                if item["ocr_error"]:
                    st.error(f"❌ Lỗi OCR: {item['ocr_error']}")
                if item["ocr_result"]:
                    result = item["ocr_result"]
                    meta1, meta2, meta3 = st.columns(3)
                    meta1.metric("Hướng chữ", result["text_direction"])
                    meta2.metric("Furigana", "Có" if result["has_furigana"] else "Không")
                    meta3.metric("Độ tin cậy", result["confidence"])
                    item["edited_text"] = st.text_area(
                        "Văn bản ảnh này (có thể chỉnh sửa):",
                        value=item["edited_text"],
                        height=180,
                        key=f"text_{item['id']}",
                    )
                    ocr_cost = estimate_cost(result.get("usage"), GEMINI_MODEL_VISION, billing_tier)
                    with st.expander("💰 Chi phí OCR ảnh này"):
                        cost1, cost2, cost3 = st.columns(3)
                        cost1.metric("Input token", f"{ocr_cost['input_tokens']:,}")
                        cost2.metric("Output token", f"{ocr_cost['output_tokens']:,}")
                        cost3.metric("Ước tính", format_cost(ocr_cost["total_cost_usd"], usd_to_jpy))
                        if billing_tier == "free":
                            st.caption(
                                "Free Tier: $0. Giá trị tương đương Paid Tier: "
                                + format_cost(ocr_cost["paid_equivalent_usd"], usd_to_jpy)
                            )

        st.divider()
        st.subheader("🧠 Phân tích theo từng trang")
        analysis_text = combined_text(items)
        pages_to_analyze = analysis_pages(items)
        partial = st.session_state.partial_page_analyses
        if not analysis_text:
            st.warning("Chưa có văn bản OCR. Hãy OCR ít nhất một ảnh trước khi phân tích.")
        else:
            with st.expander("Xem văn bản OCR theo thứ tự trang"):
                st.text_area("Nội dung gộp theo thứ tự ảnh", value=analysis_text, height=260, disabled=True)
                st.caption(
                    "Khi bấm phân tích, app sẽ gọi Gemini riêng cho từng trang/ảnh rồi mới tổng hợp. "
                    "Cách này tránh việc file nhiều trang bị dồn quá tải và chỉ phân tích trang đầu."
                )

            # Automatically resume if there is a partial analysis from a previous interrupted run.
            done_page_indices = {p["page_index"] for p in partial}
            remaining_pages = [p for p in pages_to_analyze if p["page_index"] not in done_page_indices]
            if partial and remaining_pages:
                st.info(
                    f"🔄 Phát hiện phân tích trước đó bị gián đoạn. Đang tự động chạy tiếp {len(remaining_pages)} trang còn lại..."
                )
                if st.button("🔁 Hủy và Phân tích lại từ đầu", width="stretch"):
                    st.session_state.partial_page_analyses = []
                    _persist_analysis()
                    st.rerun()
                    
                try:
                    progress = st.progress(
                        len(partial) / len(pages_to_analyze),
                        text=f"Đang tự động phân tích tiếp ({len(partial)}/{len(pages_to_analyze)})...",
                    )

                    def _resume_cb(done: int, total: int, name: str) -> None:
                        overall = len(partial) + done
                        progress.progress(
                            overall / len(pages_to_analyze),
                            text=f"Đã phân tích {overall}/{len(pages_to_analyze)}: {name}",
                        )

                    def _resume_page_done(page_result: dict) -> None:
                        st.session_state.partial_page_analyses = list(partial) + [page_result]
                        _persist_analysis()

                    new_results = text_analyzer.run_page_analyses(
                        remaining_pages,
                        analysis_language=analysis_language,
                        progress_callback=_resume_cb,
                        page_done_callback=_resume_page_done,
                    )
                    all_page_analyses = partial + new_results.get("page_analyses", [])
                    st.session_state.analysis = text_analyzer.merge_page_analyses(
                        all_page_analyses, analysis_language=analysis_language,
                    )
                    st.session_state.partial_page_analyses = []
                    _persist_analysis()
                    progress.empty()
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Lỗi tự động phân tích tiếp tục: {exc}")

            if st.button("🧠 Phân tích từng trang đã OCR", type="primary", width="stretch"):
                st.session_state.partial_page_analyses = []
                try:
                    progress = st.progress(0, text=f"Đang phân tích chi tiết {len(pages_to_analyze)} trang/ảnh...")

                    def _progress_cb(done: int, total: int, name: str) -> None:
                        progress.progress(done / total, text=f"Đã phân tích {done}/{total}: {name}")

                    def _page_done(page_result: dict) -> None:
                        st.session_state.partial_page_analyses = list(
                            st.session_state.partial_page_analyses
                        ) + [page_result]
                        _persist_analysis()

                    st.session_state.analysis = text_analyzer.run_page_analyses(
                        pages_to_analyze,
                        analysis_language=analysis_language,
                        progress_callback=_progress_cb,
                        page_done_callback=_page_done,
                    )
                    st.session_state.partial_page_analyses = []
                    _persist_analysis()
                    progress.empty()
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Lỗi phân tích: {exc}")

        if st.session_state.analysis:
            analysis = st.session_state.analysis
            result_language = analysis.get("analysis_language", analysis_language)
            ocr_costs = [
                estimate_cost(item["ocr_result"].get("usage"), GEMINI_MODEL_VISION, billing_tier)
                for item in items
                if item.get("ocr_result")
            ]
            analysis_cost = estimate_cost(analysis.get("usage"), GEMINI_MODEL_TEXT, billing_tier)
            session_cost = sum_costs([*ocr_costs, analysis_cost])
            budget = current_budget_status(api_budget_jpy, api_spent_before_jpy, session_cost["total_cost_usd"], usd_to_jpy)
            with st.expander("💰 Tổng chi phí phiên phân tích", expanded=True):
                cost1, cost2, cost3, cost4 = st.columns(4)
                cost1.metric("OCR ảnh", format_cost(sum(float(cost["total_cost_usd"]) for cost in ocr_costs), usd_to_jpy))
                cost2.metric("Phân tích văn bản", format_cost(analysis_cost["total_cost_usd"], usd_to_jpy))
                cost3.metric("Tổng token", f"{session_cost['input_tokens'] + session_cost['output_tokens']:,}")
                cost4.metric("Tổng ước tính", format_cost(session_cost["total_cost_usd"], usd_to_jpy))
                if api_budget_jpy > 0:
                    st.divider()
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Ngân sách API", f"¥{budget['budget_jpy']:,.0f} JPY")
                    b2.metric("Đã dùng trước đó", f"¥{budget['spent_before_jpy']:,.0f} JPY")
                    b3.metric("Phiên này", f"¥{budget['session_spent_jpy']:,.0f} JPY")
                    b4.metric("Ước tính còn lại", f"¥{budget['remaining_jpy']:,.0f} JPY")
                    st.progress(min(1.0, budget["used_ratio"]), text=f"Đã dùng khoảng {budget['used_ratio'] * 100:.1f}% ngân sách")
                    if budget["remaining_jpy"] <= 0:
                        st.error("Ngân sách ước tính đã hết hoặc vượt mức. Hãy kiểm tra Google Billing trước khi tiếp tục dùng API.")
                    elif budget["remaining_ratio"] <= 0.2:
                        st.warning("Ngân sách ước tính còn dưới 20%. Nên nạp thêm hoặc giảm số lần phân tích.")
                    st.caption("Theo dõi này dựa trên chi phí ước tính từ token app ghi nhận; Google Billing có thể chênh lệch nhẹ.")
                if billing_tier == "free":
                    st.caption(
                        "Free Tier hiện tính $0. Giá trị tương đương Paid Tier: "
                        + format_cost(session_cost["paid_equivalent_usd"], usd_to_jpy)
                    )
            with st.expander("💾 Lưu kết quả phân tích", expanded=True):
                st.caption("Tải file về máy/điện thoại để xem lại sau. JSON lưu dữ liệu có cấu trúc, không nhúng ảnh gốc.")
                export_stem = safe_export_stem(
                    st.text_input("Tên file lưu:", value=default_export_stem(items), key="analysis_export_stem")
                )
                docx_name = f"{export_stem}.docx"
                md_name = f"{export_stem}.md"
                json_name = f"{export_stem}.json"
                docx_bytes = export_to_docx(analysis["full_markdown"], docx_name)
                json_bytes = analysis_json_bytes(items, analysis, session_cost, billing_tier, usd_to_jpy, budget)
                save_col1, save_col2, save_col3 = st.columns(3)
                save_col1.download_button(
                    "⬇️ Word .docx",
                    data=docx_bytes,
                    file_name=docx_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    width="stretch",
                )
                save_col2.download_button(
                    "⬇️ Markdown .md",
                    data=markdown_bytes(analysis),
                    file_name=md_name,
                    mime="text/markdown",
                    width="stretch",
                )
                save_col3.download_button(
                    "⬇️ Dữ liệu .json",
                    data=json_bytes,
                    file_name=json_name,
                    mime="application/json",
                    width="stretch",
                )
            page_analyses = analysis.get("page_analyses") or [analysis]
            tab_titles = [page.get("source_label") or page.get("page_name") or f"Trang {i+1}" for i, page in enumerate(page_analyses)]
            
            tabs = st.tabs(tab_titles)
            
            for index, (tab, page) in enumerate(zip(tabs, page_analyses)):
                with tab:
                    st.subheader("📝 Tóm tắt nội dung")
                    st.write(page.get("confirmed_text") or "Không có văn bản xác nhận.")
                    st.info(page.get("summary") or "Không có tóm tắt.")
                    
                    st.subheader("📊 Từ vựng")
                    if page.get("vocabulary_all"):
                        st.dataframe(display_rows(page.get("vocabulary_all", []), result_language), width="stretch")
                    render_important_vocabulary(page.get("vocabulary_important", []))
                    
                    if result_language == "japanese":
                        st.subheader("漢字 Kanji")
                        if page.get("kanji_analysis"):
                            st.dataframe(display_rows(page.get("kanji_analysis", []), result_language), width="stretch")
                        else:
                            st.info("Trang này chưa có dữ liệu Kanji riêng.")
                    else:
                        st.subheader("🔗 Cụm từ & thành ngữ")
                        if page.get("phrasal_collocations"):
                            st.dataframe(display_rows(page.get("phrasal_collocations", []), result_language), width="stretch")
                        else:
                            st.info("Trang này chưa có cụm động từ/collocation riêng.")
                            
                    st.subheader("🔗 Từ nối" if result_language == "japanese" else "🔗 Từ nối & dấu hiệu diễn ngôn")
                    page_marker_key = "connectors" if result_language == "japanese" else "discourse_markers"
                    if page.get(page_marker_key):
                        st.dataframe(display_rows(page.get(page_marker_key, []), result_language), width="stretch")
                    else:
                        st.info("Trang này chưa có dữ liệu từ nối riêng.")
                        
                    st.subheader("🧩 Ngữ pháp")
                    render_grammar_points(page.get("grammar_points", []))
                    
                    st.subheader("🔎 Mẫu câu")
                    if page.get("sentence_patterns"):
                        for pattern in page.get("sentence_patterns", []):
                            with st.expander(f"🔎 {pattern.get('pattern', 'Mẫu câu')}"):
                                if pattern.get("example"):
                                    st.markdown("**Ví dụ trong bài:**")
                                    st.code(pattern["example"], language=None)
                                if pattern.get("explanation"):
                                    st.markdown(f"**Giải thích:** {pattern['explanation']}")
                    else:
                        st.info("Trang này chưa có mẫu câu riêng.")
                        
                    with st.expander("💾 Xem Markdown đầy đủ của trang"):
                        st.markdown(page.get("full_markdown") or "Không có dữ liệu.")

with tab_url:
    st.subheader("🌐 Phân Tích Bài Báo Từ URL")
    st.caption("Hỗ trợ: NHK, Asahi, Mainichi, BBC, Reuters, Japan Times và hầu hết báo lớn")

    url_input = st.text_input(
        "Dán link bài báo:",
        placeholder="https://www3.nhk.or.jp/news/html/...",
        key="url_input_field",
    )

    col_btn1, col_btn2, col_space = st.columns([1.2, 1.5, 4])
    with col_btn1:
        btn_fetch = st.button("📥 Tải bài báo", key="btn_fetch_url",
                              disabled=not url_input)
    with col_btn2:
        btn_analyze = st.button("🔍 Phân tích", key="btn_analyze_url",
                                disabled=not st.session_state.url_scrape_result)

    # --- BƯỚC 1: Scrape bài báo ---
    if btn_fetch and url_input:
        with st.spinner("Đang tải nội dung bài báo..."):
            try:
                scraped = fetch_article(url_input.strip())
                st.session_state.url_scrape_result = scraped
                st.session_state.url_analysis_result = None
                lang_label = "🇯🇵 Tiếng Nhật" if scraped["lang"] == "ja" else "🇬🇧 Tiếng Anh"
                st.success(f"Tải thành công! Ngôn ngữ phát hiện: {lang_label} | {scraped['word_count']} từ")
            except Exception as e:
                st.error(f"Lỗi: {e}")

    # --- Preview nội dung ---
    if st.session_state.url_scrape_result:
        r = st.session_state.url_scrape_result
        lang_badge = "🇯🇵 Tiếng Nhật" if r["lang"] == "ja" else "🇬🇧 Tiếng Anh"
        st.markdown(f"**📰 Tiêu đề:** {r['title']}")
        st.markdown(f"**🔗 Nguồn:** {r['source_url']}")
        st.markdown(f"**🌍 Ngôn ngữ:** {lang_badge} | **Số từ:** {r['word_count']} | **Ký tự:** {r['char_count']}")

        with st.expander("📄 Xem nội dung đã trích xuất (có thể chỉnh sửa trước khi phân tích)"):
            edited_text = st.text_area(
                "Nội dung bài báo:",
                value=r["clean_text"],
                height=300,
                key="url_editable_text",
            )
            if edited_text != r["clean_text"]:
                if st.button("💾 Lưu chỉnh sửa", key="btn_save_edit"):
                    st.session_state.url_scrape_result["clean_text"] = edited_text
                    st.session_state.url_scrape_result["word_count"] = len(edited_text.split())
                    st.success("Đã lưu!")

    # --- BƯỚC 2: Phân tích ---
    if btn_analyze and st.session_state.url_scrape_result:
        r = st.session_state.url_scrape_result
        with st.spinner("Đang phân tích văn bản... (có thể mất 30–60 giây)"):
            try:
                url_lang = "japanese" if r["lang"] == "ja" else "english"
                url_analysis = text_analyzer.run_analysis(
                    r["clean_text"], [], analysis_language=url_lang
                )
                url_analysis["_source"] = r
                st.session_state.url_analysis_result = url_analysis
                st.success("Phân tích hoàn tất!")
            except Exception as e:
                st.error(f"Lỗi phân tích: {e}")

    # --- Hiển thị kết quả ---
    if st.session_state.url_analysis_result:
        url_result = st.session_state.url_analysis_result
        url_src = url_result.get("_source", {})
        url_result_language = "japanese" if url_src.get("lang") == "ja" else "english"

        st.divider()
        st.markdown("### 📊 Báo Cáo Phân Tích")
        st.markdown(f"**Nguồn:** [{url_src.get('title', url_src.get('source_url', ''))}]({url_src.get('source_url', '')})")
        st.markdown(f"**Tóm tắt:** {url_result.get('summary', '')}")

        st.subheader("📝 Nội dung xác nhận")
        st.write(url_result.get("confirmed_text") or "Không có văn bản xác nhận.")
        st.info(url_result.get("summary") or "Không có tóm tắt.")

        st.subheader("📊 Từ vựng")
        if url_result.get("vocabulary_all"):
            st.dataframe(display_rows(url_result.get("vocabulary_all", []), url_result_language), width="stretch")
        render_important_vocabulary(url_result.get("vocabulary_important", []))

        if url_result_language == "japanese":
            st.subheader("漢字 Kanji")
            if url_result.get("kanji_analysis"):
                st.dataframe(display_rows(url_result.get("kanji_analysis", []), url_result_language), width="stretch")
            else:
                st.info("Chưa có dữ liệu Kanji.")
        else:
            st.subheader("🔗 Cụm từ & thành ngữ")
            if url_result.get("phrasal_collocations"):
                st.dataframe(display_rows(url_result.get("phrasal_collocations", []), url_result_language), width="stretch")
            else:
                st.info("Chưa có cụm động từ/collocation.")

        st.subheader("🔗 Từ nối" if url_result_language == "japanese" else "🔗 Từ nối & dấu hiệu diễn ngôn")
        url_marker_key = "connectors" if url_result_language == "japanese" else "discourse_markers"
        if url_result.get(url_marker_key):
            st.dataframe(display_rows(url_result.get(url_marker_key, []), url_result_language), width="stretch")
        else:
            st.info("Chưa có dữ liệu từ nối.")

        st.subheader("🧩 Ngữ pháp")
        render_grammar_points(url_result.get("grammar_points", []))

        st.subheader("🔎 Mẫu câu")
        if url_result.get("sentence_patterns"):
            for pattern in url_result.get("sentence_patterns", []):
                with st.expander(f"🔎 {pattern.get('pattern', 'Mẫu câu')}"):
                    if pattern.get("example"):
                        st.markdown("**Ví dụ trong bài:**")
                        st.code(pattern["example"], language=None)
                    if pattern.get("explanation"):
                        st.markdown(f"**Giải thích:** {pattern['explanation']}")
        else:
            st.info("Chưa có mẫu câu.")

        with st.expander("💾 Xem Markdown đầy đủ"):
            st.markdown(url_result.get("full_markdown") or "Không có dữ liệu.")
