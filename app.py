"""Streamlit entry point for Japanese OCR Analyzer."""

from __future__ import annotations

import streamlit as st

from config import GEMINI_MODEL_TEXT, GEMINI_MODEL_VISION, MAX_IMAGE_SIZE_MB, SUPPORTED_FORMATS
from modules.cost_estimator import estimate_cost, format_cost, sum_costs
from modules.doc_exporter import export_to_docx
from modules.multi_image_workflow import add_image_items, combined_notes, combined_text, move_image_item
from modules.ocr_engine import run_ocr
from modules.text_analyzer import run_analysis


st.set_page_config(page_title="Japanese OCR Analyzer", page_icon="🔍", layout="wide")
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
    "uploader_version": 0,
    "camera_version": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def clear_analysis() -> None:
    st.session_state.analysis = None


def add_sources(sources: list[tuple[str, bytes]]) -> None:
    items, added, errors = add_image_items(st.session_state.image_items, sources)
    st.session_state.image_items = items
    if added:
        clear_analysis()
        st.toast(f"Đã thêm {len(added)} ảnh.", icon="✅")
    for error in errors:
        st.error(f"❌ Không thể thêm ảnh: {error}")


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


st.title("🔍 Japanese OCR Analyzer")
st.caption("Tải nhiều ảnh, OCR từng ảnh hoặc toàn bộ, rồi gộp nội dung để phân tích chung.")

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
st.sidebar.caption(f"Kích thước tối đa mỗi ảnh: {MAX_IMAGE_SIZE_MB} MB")
st.sidebar.subheader("💰 Ước tính chi phí")
billing_tier = st.sidebar.radio(
    "Gói Gemini",
    options=["free", "paid"],
    format_func=lambda value: "Free Tier" if value == "free" else "Paid Tier (Standard)",
    horizontal=True,
)
usd_to_vnd = st.sidebar.number_input("Tỷ giá USD → VND", min_value=1.0, value=25_500.0, step=100.0)
st.sidebar.caption("Gemini 2.5 Flash: input $0.30/M token, output $2.50/M token.")
st.sidebar.markdown("[Xem bảng giá Gemini chính thức](https://ai.google.dev/gemini-api/docs/pricing)")
with st.sidebar.expander("💡 Luồng sử dụng"):
    st.write("1. Thêm một hoặc nhiều ảnh.")
    st.write("2. OCR từng ảnh hoặc OCR toàn bộ.")
    st.write("3. Chỉnh văn bản của từng ảnh.")
    st.write("4. Phân tích chung tất cả ảnh đã OCR.")

with st.expander("➕ Thêm ảnh vào bộ phân tích", expanded=not items):
    upload_tab, camera_tab = st.tabs(["📁 Chọn nhiều ảnh từ máy", "📷 Chụp thêm ảnh"])
    with upload_tab:
        uploaded_files = st.file_uploader(
            "Kéo thả nhiều ảnh hoặc bấm nút để chọn file",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key=f"multi_uploader_{st.session_state.uploader_version}",
            help=f"Mỗi ảnh tối đa {MAX_IMAGE_SIZE_MB} MB.",
        )
        if uploaded_files and st.button("➕ Thêm ảnh đã chọn", type="primary", width="stretch"):
            add_sources([(file.name, file.getvalue()) for file in uploaded_files])
            st.session_state.uploader_version += 1
            st.rerun()
    with camera_tab:
        camera_file = st.camera_input(
            "Chụp ảnh văn bản tiếng Nhật",
            key=f"camera_{st.session_state.camera_version}",
        )
        if camera_file and st.button("➕ Thêm ảnh vừa chụp", width="stretch"):
            add_sources([(f"camera_{len(items) + 1}.jpg", camera_file.getvalue())])
            st.session_state.camera_version += 1
            st.rerun()

if not items:
    st.info("Hãy thêm một hoặc nhiều ảnh để bắt đầu.")
    st.stop()

st.subheader(f"Ảnh trong bộ phân tích ({len(items)})")
controls_left, controls_right = st.columns(2)
with controls_left:
    if st.button("🔍 OCR tất cả ảnh chưa xử lý", type="primary", width="stretch"):
        pending = [item for item in items if not item["ocr_result"]]
        progress = st.progress(0, text="Đang OCR...")
        for index, item in enumerate(pending, 1):
            progress.progress(index / len(pending), text=f"Đang OCR: {item['name']}")
            run_item_ocr(item)
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
                cost3.metric("Ước tính", format_cost(ocr_cost["total_cost_usd"], usd_to_vnd))
                if billing_tier == "free":
                    st.caption(
                        "Free Tier: $0. Giá trị tương đương Paid Tier: "
                        + format_cost(ocr_cost["paid_equivalent_usd"], usd_to_vnd)
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
                st.session_state.analysis = run_analysis(analysis_text, combined_notes(items))
        except Exception as exc:
            st.error(f"❌ Lỗi phân tích: {exc}")

if st.session_state.analysis:
    analysis = st.session_state.analysis
    ocr_costs = [
        estimate_cost(item["ocr_result"].get("usage"), GEMINI_MODEL_VISION, billing_tier)
        for item in items
        if item.get("ocr_result")
    ]
    analysis_cost = estimate_cost(analysis.get("usage"), GEMINI_MODEL_TEXT, billing_tier)
    session_cost = sum_costs([*ocr_costs, analysis_cost])
    with st.expander("💰 Tổng chi phí phiên phân tích", expanded=True):
        cost1, cost2, cost3, cost4 = st.columns(4)
        cost1.metric("OCR ảnh", format_cost(sum(float(cost["total_cost_usd"]) for cost in ocr_costs), usd_to_vnd))
        cost2.metric("Phân tích văn bản", format_cost(analysis_cost["total_cost_usd"], usd_to_vnd))
        cost3.metric("Tổng token", f"{session_cost['input_tokens'] + session_cost['output_tokens']:,}")
        cost4.metric("Tổng ước tính", format_cost(session_cost["total_cost_usd"], usd_to_vnd))
        if billing_tier == "free":
            st.caption(
                "Free Tier hiện tính $0. Giá trị tương đương Paid Tier: "
                + format_cost(session_cost["paid_equivalent_usd"], usd_to_vnd)
            )
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📝 Tóm tắt", "📊 Từ vựng", "漢字 Kanji", "🔗 Từ nối & Ngữ pháp", "💾 Xuất Word"]
    )
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
        st.dataframe(analysis["kanji_analysis"], width="stretch")
    with tab4:
        st.subheader("Từ nối câu")
        st.dataframe(analysis["connectors"], width="stretch")
        st.subheader("Điểm ngữ pháp")
        for point in analysis["grammar_points"]:
            with st.expander(f"📌 {point['name']}"):
                st.write(f"**Quy tắc:** {point['rule']}")
                st.code(point["example"], language=None)
                st.write(f"**Giải thích:** {point['explanation']}")
    with tab5:
        filename = st.text_input("Tên file:", value="japanese_multi_image_analysis.docx")
        docx_bytes = export_to_docx(analysis["full_markdown"], filename)
        st.download_button(
            "⬇️ Tải xuống .docx",
            data=docx_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        st.markdown(analysis["full_markdown"])
