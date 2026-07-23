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

from modules.dialogue_generator import generate_dialogue, suggest_topics



import subprocess

import sys as _sys

import json

from pathlib import Path

from modules.job_store import create_job, get_job







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

    "current_job_id": None,

    "recent_topics": [],

    "dialogue_result": None,

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





_WORKER_PATH = str(Path(__file__).resolve().parent / "worker.py")

_PROJECT_DIR = str(Path(__file__).resolve().parent)





# ── Check background job status if job_id is present ─────────────────────

job_id_from_url = st.query_params.get("job_id")

if job_id_from_url:

    job = get_job(job_id_from_url)

    if job:

        if job["status"] == "done":

            st.success("Phân tích đã hoàn tất!")

            if job["lang"] in ("pdf_ja", "pdf_en", "ai_ja", "ai_en"):

                if st.session_state.analysis is None:

                    st.session_state.analysis = job["result"]

                    _persist_analysis()

        elif job["status"] == "running":

            st.warning("⏳ Đang phân tích trong nền, vui lòng tải lại trang sau vài giây...")

            st.button("🔄 Tải lại")

        elif job["status"] == "failed":

            st.error(f"Phân tích thất bại: {job['error']}")

            if st.button("🔄 Thử lại"):

                del st.query_params["job_id"]

                st.rerun()

        else:

            st.info("⏳ Đang chờ xử lý...")

            st.button("🔄 Tải lại")





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

    if "job_id" in st.query_params:

        del st.query_params["job_id"]

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





def run_item_ocr(item: dict, model_name: str | None = None) -> None:

    item["ocr_error"] = None

    try:

        result = run_ocr(item["processed_image_bytes"], item["report"], model_name=model_name)

        item["ocr_result"] = result

        item["edited_text"] = result["clean_text"]

    except Exception as exc:

        item["ocr_error"] = str(exc)











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

st.sidebar.subheader("🤖 Cấu hình AI Model")

ocr_model_choice = st.sidebar.selectbox(
    "Model OCR (Vision)",
    options=["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview", "gemini-2.5-flash"],
    index=0,
    help="Model được dùng để đọc chữ từ ảnh/PDF (Mặc định: gemini-3.5-flash)",
)

text_model_choice = st.sidebar.selectbox(
    "Model Phân tích văn bản",
    options=["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview", "gemini-2.5-flash"],
    index=0,
    help="Model được dùng để dịch thuật và giải thích ngữ pháp (Mặc định: gemini-3.5-flash)",
)

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

st.sidebar.caption("Gemini 3.5 Flash: input $0.30/M token, output $2.50/M token.")

st.sidebar.markdown("[Xem bảng giá Gemini chính thức](https://ai.google.dev/gemini-api/docs/pricing)")



with st.sidebar.expander("💡 Luồng sử dụng"):

    st.write("1. Thêm ảnh hoặc PDF từ máy/ứng dụng Files.")

    st.write("2. Mỗi trang PDF sẽ được chuyển thành một ảnh.")

    st.write("3. OCR từng trang hoặc OCR toàn bộ.")

    st.write("4. Phân tích chung tất cả nội dung đã OCR.")



tab_ocr, tab_dialogue = st.tabs(["📷 Phân tích từ Ảnh / PDF", "💬 Luyện Hội Thoại"])



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

                        model_name=text_model_choice,

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



            if analysis_language == "japanese":
                if st.button("🧠 Phân tích bằng Gemini", type="primary", width="stretch"):
                    st.session_state.partial_page_analyses = []
                    _persist_analysis()
                    detected_lang = "pdf_ja"
                    input_data = {
                        "pages": pages_to_analyze,
                        "model_name": text_model_choice,
                    }
                    input_text = json.dumps(input_data)
                    job_id = create_job(input_text, detected_lang)
                    subprocess.Popen(
                        [_sys.executable, _WORKER_PATH, job_id],
                        stdout=subprocess.DEVNULL,
                        stderr=open(_PROJECT_DIR + "/worker_error.log", "a"),
                        cwd=_PROJECT_DIR,
                    )
                    st.session_state.current_job_id = job_id
                    st.query_params["job_id"] = job_id
                    st.success(f"Đã bắt đầu phân tích! Job ID: {job_id}")
                    st.info("Bạn có thể đóng tab này — kết quả sẽ được lưu lại. Mở lại link có job_id để xem kết quả.")
                    st.rerun()
            else:
                if st.button("🧠 Phân tích từng trang đã OCR", type="primary", width="stretch"):
                    st.session_state.partial_page_analyses = []
                    _persist_analysis()
                    detected_lang = "pdf_en"
                    input_data = {
                        "pages": pages_to_analyze,
                        "model_name": text_model_choice,
                    }
                    input_text = json.dumps(input_data)
                    job_id = create_job(input_text, detected_lang)
                    subprocess.Popen(
                        [_sys.executable, _WORKER_PATH, job_id],
                        stdout=subprocess.DEVNULL,
                        stderr=open(_PROJECT_DIR + "/worker_error.log", "a"),
                        cwd=_PROJECT_DIR,
                    )
                    st.session_state.current_job_id = job_id
                    st.query_params["job_id"] = job_id
                    st.success(f"Đã bắt đầu phân tích! Job ID: {job_id}")
                    st.info("Bạn có thể đóng tab này — kết quả sẽ được lưu lại. Mở lại link có job_id để xem kết quả.")
                    st.rerun()



        if st.session_state.analysis:

            analysis = st.session_state.analysis

            result_language = analysis.get("analysis_language", analysis_language)

            ocr_costs = [

                estimate_cost(item["ocr_result"].get("usage"), ocr_model_choice, billing_tier)

                for item in items

                if item.get("ocr_result")

            ]

            analysis_cost = estimate_cost(analysis.get("usage"), text_model_choice, billing_tier)

            session_cost = sum_costs([*ocr_costs, analysis_cost])

            with st.expander("💰 Tổng chi phí phiên phân tích", expanded=True):

                cost1, cost2, cost3, cost4 = st.columns(4)

                cost1.metric("OCR ảnh", format_cost(sum(float(cost["total_cost_usd"]) for cost in ocr_costs), usd_to_jpy))

                cost2.metric("Phân tích văn bản", format_cost(analysis_cost["total_cost_usd"], usd_to_jpy))

                cost3.metric("Tổng token", f"{session_cost['input_tokens'] + session_cost['output_tokens']:,}")

                cost4.metric("Tổng ước tính", format_cost(session_cost["total_cost_usd"], usd_to_jpy))

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

                json_bytes = analysis_json_bytes(items, analysis, session_cost, billing_tier, usd_to_jpy)

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



            # ── AI Quality Review expander ──────────────────────────────────

            _pipeline_meta = analysis.get("_pipeline_result")

            if _pipeline_meta:

                with st.expander("🔍 Kiểm tra chất lượng AI", expanded=False):

                    _qs = _pipeline_meta.get("quality_status", "unknown")

                    _status_labels = {

                        "verified": "✅ Verified — Gemini xác nhận chính xác",

                        "corrected": "🔧 Corrected — Gemini đã sửa lỗi",

                        "review_unavailable": "⚠️ Review không khả dụng",

                    }

                    st.info(_status_labels.get(_qs, _qs))



                    _review = _pipeline_meta.get("review", {})

                    if _review.get("review_note_vi"):

                        st.write(f"**Ghi chú Gemini:** {_review['review_note_vi']}")



                    _issues = _review.get("issues", [])

                    if _issues:

                        st.write(f"**Số lỗi Gemini phát hiện:** {len(_issues)}")

                        for _iss in _issues:

                            st.caption(

                                f"• `{_iss.get('item_id', '')}` — {_iss.get('problem_vi', '')} "

                                f"(action: {_iss.get('action', '')})"

                            )



                    _warnings = _pipeline_meta.get("warnings", [])

                    if _warnings:

                        st.write("**Cảnh báo khi áp dụng sửa lỗi:**")

                        for _w in _warnings:

                            st.caption(f"⚠️ {_w}")



                    _missing = _review.get("missing_items", [])

                    if _missing:

                        st.write("**Mục Gemini đề nghị xem thêm:**")

                        for _m in _missing:

                            st.caption(

                                f"• [{_m.get('category', '')}] {_m.get('term_or_name', '')} — {_m.get('reason_vi', '')}"

                            )



with tab_dialogue:

    st.subheader("💬 Luyện Hội Thoại Hằng Ngày")



    col1, col2 = st.columns(2)

    with col1:

        dlg_language = st.selectbox("Ngôn ngữ:", ["Tiếng Nhật", "Tiếng Anh"], key="dlg_lang")

    with col2:

        dlg_level = st.selectbox("Cấp độ:", ["Sơ cấp", "Trung cấp", "Cao cấp"], key="dlg_level")



    if st.button("🎲 Gợi ý chủ đề hôm nay", key="btn_suggest_topic"):

        with st.spinner("Đang tìm chủ đề..."):

            topics = suggest_topics(dlg_language, dlg_level, st.session_state.recent_topics)

            st.session_state.suggested_topics = topics



    if st.session_state.get("suggested_topics"):

        for t in st.session_state.suggested_topics:

            if st.button(f"📌 {t['topic']}", key=f"topic_{t['topic']}"):

                st.session_state.selected_topic = t["topic"]



    topic_input = st.text_input(

        "Hoặc nhập chủ đề của riêng bạn:",

        value=st.session_state.get("selected_topic", ""),

        key="dlg_topic_input",

    )



    vocab_input = st.text_area(

        "Từ vựng muốn luyện (mỗi từ 1 dòng, có thể để trống):",

        key="dlg_vocab_input", height=80,

    )

    grammar_input = st.text_area(

        "Cấu trúc ngữ pháp muốn luyện (mỗi cấu trúc 1 dòng, có thể để trống):",

        key="dlg_grammar_input", height=80,

    )



    if st.button("✨ Tạo hội thoại", key="btn_generate_dialogue", disabled=not topic_input):

        vocab_list = [v.strip() for v in vocab_input.splitlines() if v.strip()]

        grammar_list = [g.strip() for g in grammar_input.splitlines() if g.strip()]

        with st.spinner("Đang tạo hội thoại..."):

            try:

                result = generate_dialogue(

                    topic_input, dlg_language, vocab_list, grammar_list, dlg_level

                )

                st.session_state.dialogue_result = result

                st.session_state.recent_topics.append(topic_input)

                if not result["fully_covered"]:

                    st.warning("Một số từ/ngữ pháp chưa được dùng hết, nhưng đây là bản tốt nhất.")

            except Exception as e:

                st.error(f"Lỗi: {e}")



    if st.session_state.dialogue_result:

        r = st.session_state.dialogue_result

        st.markdown(f"### 📖 Chủ đề: {r['topic']}")

        for turn in r["dialogue"]:

            icon = "🗣️" if turn["speaker"] == "A" else "💭"

            st.markdown(f"**{icon} {turn['speaker']}:** {turn['text']}")

            if turn.get("text_hira"):

                st.caption(f"_{turn['text_hira']}_")

            st.caption(turn["text_vi"])

            if turn["highlights"]:

                st.caption(f"🎯 Dùng: {', '.join(turn['highlights'])}")



        st.divider()

        if r.get("summary"):

            st.markdown("#### 📚 Tóm tắt Từ vựng & Ngữ pháp")

            st.info(r["summary"])



        with st.expander("✅ Kiểm tra độ phủ từ vựng/ngữ pháp"):

            for target, covered in r["coverage_check"].items():

                icon = "✅" if covered else "❌"

                st.markdown(f"{icon} {target}")



        if r["notes"]:

            st.info(r["notes"])

