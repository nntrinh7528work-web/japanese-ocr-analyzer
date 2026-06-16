"""Streamlit entry point for Japanese/English OCR Analyzer."""

from __future__ import annotations

import importlib

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
from modules.multi_image_workflow import add_upload_items, combined_notes, combined_text, move_image_item
from modules.ocr_engine import run_ocr
from modules.result_exporter import analysis_json_bytes, default_export_stem, markdown_bytes, safe_export_stem
import modules.text_analyzer as text_analyzer


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
    "upload_messages": [],
    "upload_errors": [],
    "uploader_version": 0,
    "camera_version": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def clear_analysis() -> None:
    st.session_state.analysis = None


def add_sources(sources: list[tuple[str, bytes]]) -> bool:
    items, added, errors = add_upload_items(st.session_state.image_items, sources)
    st.session_state.image_items = items
    st.session_state.upload_messages = []
    st.session_state.upload_errors = errors
    if added:
        clear_analysis()
        st.session_state.upload_messages.append(f"Đã thêm {len(added)} ảnh/trang PDF.")
    return bool(added)


def remove_image(item_id: str) -> None:
    st.session_state.image_items = [item for item in st.session_state.image_items if item["id"] != item_id]
    clear_analysis()


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
    st.rerun()
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
    st.stop()

st.subheader(f"Ảnh/trang PDF trong bộ phân tích ({len(items)})")
controls_left, controls_right = st.columns(2)
with controls_left:
    if st.button("🔍 OCR tất cả ảnh chưa xử lý", type="primary", width="stretch"):
        pending = [item for item in items if not item["ocr_result"]]
        progress = st.progress(0, text="Đang OCR...")
        for index, item in enumerate(pending, 1):
            progress.progress(index / len(pending), text=f"Đang OCR: {item['name']}")
            run_item_ocr(item)
            st.session_state.image_items = items
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
st.subheader("🧠 Phân tích chung nhiều ảnh")
analysis_text = combined_text(items)
if not analysis_text:
    st.warning("Chưa có văn bản OCR. Hãy OCR ít nhất một ảnh trước khi phân tích.")
else:
    with st.expander("Xem văn bản sẽ được gộp để phân tích"):
        st.text_area("Nội dung gộp theo thứ tự ảnh", value=analysis_text, height=260, disabled=True)
    if st.button("🧠 Phân tích tất cả ảnh đã OCR", type="primary", width="stretch"):
        try:
            with st.spinner("Đang phân tích nội dung từ nhiều ảnh..."):
                st.session_state.analysis = text_analyzer.run_analysis(
                    analysis_text,
                    combined_notes(items),
                    analysis_language=analysis_language,
                )
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
    detail_tabs = (
        ["📝 Tóm tắt", "📊 Từ vựng", "漢字 Kanji", "🔗 Từ nối & Ngữ pháp", "💾 Xem bản lưu"]
        if result_language == "japanese"
        else ["📝 Tóm tắt", "📊 Từ vựng", "🔗 Phrasal & Collocations", "🧩 Discourse & Grammar", "💾 Xem bản lưu"]
    )
    tab1, tab2, tab3, tab4, tab5 = st.tabs(detail_tabs)
    with tab1:
        st.subheader("Văn bản đã xác nhận")
        st.write(analysis["confirmed_text"])
        st.subheader("Tóm tắt")
        st.info(analysis["summary"])
    with tab2:
        st.dataframe(analysis["vocabulary_all"], width="stretch")
        st.subheader("Từ vựng quan trọng")
        st.dataframe(analysis["vocabulary_important"], width="stretch")
    with tab3:
        if result_language == "japanese":
            if analysis["kanji_analysis"]:
                st.dataframe(analysis["kanji_analysis"], width="stretch")
            else:
                st.info("Chưa trích xuất được bảng Kanji riêng. Xem nội dung gốc bên dưới.")
                st.markdown(analysis.get("section_markdown", {}).get("kanji") or "Không có dữ liệu Kanji.")
        elif analysis.get("phrasal_collocations"):
            st.dataframe(analysis["phrasal_collocations"], width="stretch")
        else:
            st.info("Chưa trích xuất được bảng phrasal verbs/collocations riêng. Xem nội dung gốc bên dưới.")
            st.markdown(
                analysis.get("section_markdown", {}).get("phrasal_collocations")
                or "Không có dữ liệu phrasal verbs/collocations."
            )
    with tab4:
        st.subheader("Từ nối câu" if result_language == "japanese" else "Linking words & discourse markers")
        marker_key = "connectors" if result_language == "japanese" else "discourse_markers"
        marker_markdown_key = "connectors" if result_language == "japanese" else "discourse_markers"
        if analysis.get(marker_key):
            st.dataframe(analysis[marker_key], width="stretch")
        else:
            st.info("Chưa trích xuất được bảng từ nối/discourse markers riêng.")
            st.markdown(
                analysis.get("section_markdown", {}).get(marker_markdown_key)
                or "Không có dữ liệu từ nối/discourse markers."
            )
        st.subheader("Điểm ngữ pháp" if result_language == "japanese" else "Grammar points")
        if analysis["grammar_points"]:
            for point in analysis["grammar_points"]:
                with st.expander(f"📌 {point['name']}"):
                    st.write(f"**{'Quy tắc' if result_language == 'japanese' else 'Rule'}:** {point['rule']}")
                    st.code(point["example"], language=None)
                    st.write(f"**{'Giải thích' if result_language == 'japanese' else 'Explanation'}:** {point['explanation']}")
        else:
            st.info("Chưa trích xuất được mục ngữ pháp/grammar riêng. Xem nội dung gốc bên dưới.")
            st.markdown(analysis.get("section_markdown", {}).get("grammar") or "Không có dữ liệu ngữ pháp/grammar.")
    with tab5:
        st.info("Dùng mục '💾 Lưu kết quả phân tích' phía trên để tải Word, Markdown hoặc JSON.")
        st.markdown(analysis["full_markdown"])
