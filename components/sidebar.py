"""Sidebar component for Streamlit UI."""

from __future__ import annotations
import streamlit as st
from config import MAX_IMAGE_SIZE_MB, MAX_PDF_PAGES, MAX_PDF_SIZE_MB
from modules.notion_sync import notion_connection_state


def render_sidebar(
    items: list[dict],
    clear_analysis_fn,
    persist_items_fn,
    persist_analysis_fn,
) -> dict:
    """Render sidebar elements and return a configuration dictionary."""
    st.sidebar.header("📚 Bộ ảnh phân tích")
    
    col_m1, col_m2 = st.sidebar.columns(2)
    col_m1.metric("Số ảnh", len(items))
    col_m2.metric("Đã OCR", sum(bool(item.get("ocr_result")) for item in items))

    if items and st.sidebar.button("🗑️ Xóa toàn bộ bộ ảnh", use_container_width=True):
        st.session_state.image_items = []
        st.session_state.analysis = None
        st.session_state.uploader_version += 1
        st.session_state.camera_version += 1
        persist_items_fn()
        persist_analysis_fn()
        st.rerun()

    # Session code display
    if st.session_state.session_id:
        st.sidebar.divider()
        st.sidebar.subheader("💾 Phiên làm việc")
        st.sidebar.code(st.session_state.session_id, language=None)
        st.sidebar.caption(
            "Mã phiên tự động. Khi thoát app, mở lại link có mã này để khôi phục toàn bộ tiến trình. "
            "Phiên được giữ tối đa 30 ngày trên máy chủ hiện tại."
        )

    st.sidebar.divider()
    st.sidebar.header("⚙️ Settings")

    dark_mode = st.sidebar.toggle(
        "🌙 Chế độ tối (Dark Mode)",
        value=st.session_state.get("dark_mode", False),
        key="dark_mode_toggle",
    )
    st.session_state["dark_mode"] = dark_mode

    show_preprocessing = st.sidebar.toggle("Hiển thị chi tiết tiền xử lý", value=True)
    st.sidebar.caption(f"Ảnh tối đa {MAX_IMAGE_SIZE_MB} MB · PDF tối đa {MAX_PDF_SIZE_MB} MB/{MAX_PDF_PAGES} trang")

    # Group AI configuration inside expander for clean visual hierarchy
    with st.sidebar.expander("🤖 Cấu hình AI Model", expanded=True):
        ocr_model_choice = st.selectbox(
            "Model OCR (Vision)",
            options=["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview", "gemini-2.5-flash"],
            index=0,
            help="Model được dùng để đọc chữ từ ảnh/PDF (Mặc định: gemini-3.5-flash)",
        )

        text_model_choice = st.selectbox(
            "Model Phân tích văn bản",
            options=["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview", "gemini-2.5-flash"],
            index=0,
            help="Model được dùng để dịch thuật và giải thích ngữ pháp (Mặc định: gemini-3.5-flash)",
        )

        reasoning_effort = st.radio(
            "Mức độ suy luận (Reasoning)",
            options=["standard", "deep"],
            format_func=lambda val: "⚡ Tiêu chuẩn (Nhanh)" if val == "standard" else "🧠 Chuyên sâu (Deep Reasoning)",
            help="Chế độ Chuyên sâu giúp Gemini 3.5 Flash & 3.1 Pro lập luận kỹ lưỡng các sắc thái ngữ pháp, từ vựng và câu văn phức tạp.",
        )

        st.caption("Ngôn ngữ phân tích được chọn riêng ở đầu từng bài để ảnh Nhật và Anh không bị trộn.")

        analysis_mode = str(st.session_state.get("analysis_mode") or "full_analysis")

        auto_sentence_deep_dive = st.toggle(
            "Thêm giải mã sâu câu dài (tốn thêm token)",
            value=bool(st.session_state.get("auto_sentence_deep_dive", False)),
            help="Tự chọn tối đa 3 câu khó mỗi trang và 15 câu trong toàn tài liệu để phân tích sâu.",
        )
        st.session_state["auto_sentence_deep_dive"] = auto_sentence_deep_dive

        auto_translation_guidance = analysis_mode == "sentence_guidance"
        st.session_state["auto_translation_guidance"] = auto_translation_guidance
        st.caption("Chọn loại phân tích bằng hai nút riêng bên dưới phần văn bản OCR.")

    with st.sidebar.expander("🗂️ Lưu vào Notion", expanded=False):
        notion_state = notion_connection_state()
        if notion_state["configured"]:
            st.success(f"Notion: {notion_state['label']}")
            if notion_state.get("migration_error"):
                st.caption(f"Lỗi nâng cấp gần nhất: {notion_state['migration_error']}")
        else:
            st.info(f"Notion: {notion_state['label']}")
            st.caption(
                "Thiết lập một lần bằng NOTION_TOKEN và NOTION_PARENT_PAGE_ID trong "
                "Streamlit Secrets. Token không được lưu trong phiên hoặc file xuất."
            )
        auto_notion_sync = st.toggle(
            "Tự động lưu bài hoàn tất",
            value=bool(st.session_state.get("auto_notion_sync", True)),
            disabled=not notion_state["configured"],
            help="Đồng bộ trong nền sau khi toàn bộ phân tích chính hoàn tất.",
        )
        st.session_state["auto_notion_sync"] = auto_notion_sync

    # Group cost options inside expander
    with st.sidebar.expander("💰 Ước tính chi phí", expanded=False):
        billing_tier = st.radio(
            "Gói Gemini",
            options=["free", "paid"],
            index=0 if st.session_state.get("billing_tier", "free") == "free" else 1,
            format_func=lambda val: "Free Tier" if val == "free" else "Paid Tier (Standard)",
            horizontal=True,
        )
        st.session_state["billing_tier"] = billing_tier

        budget_jpy = st.number_input(
            "Ngân sách API (JPY)", min_value=0.0,
            value=float(st.session_state.get("budget_jpy", 0.0)), step=100.0,
            key="budget_jpy_input",
        )
        spent_before_jpy = st.number_input(
            "Đã chi trước phiên này (JPY)", min_value=0.0,
            value=float(st.session_state.get("spent_before_jpy", 0.0)), step=100.0,
            key="spent_before_jpy_input",
        )
        usd_to_jpy = st.number_input(
            "Tỷ giá USD → JPY", min_value=1.0,
            value=float(st.session_state.get("usd_to_jpy", 155.0)), step=1.0,
            key="usd_to_jpy_input",
        )
        st.session_state["budget_jpy"] = budget_jpy
        st.session_state["spent_before_jpy"] = spent_before_jpy
        st.session_state["usd_to_jpy"] = usd_to_jpy
        st.caption("Gemini 3.5 Flash Standard: input $1.50/M token, output $9.00/M token.")
        st.markdown("[Xem bảng giá Gemini chính thức](https://ai.google.dev/gemini-api/docs/pricing)")

    with st.sidebar.expander("💡 Luồng sử dụng"):
        st.write("1. Thêm ảnh hoặc PDF từ máy/ứng dụng Files.")
        st.write("2. Mỗi trang PDF sẽ được chuyển thành một ảnh.")
        st.write("3. OCR từng trang hoặc OCR toàn bộ.")
        st.write("4. Phân tích chung tất cả nội dung đã OCR.")

    return {
        "dark_mode": dark_mode,
        "show_preprocessing": show_preprocessing,
        "ocr_model_choice": ocr_model_choice,
        "text_model_choice": text_model_choice,
        "reasoning_effort": reasoning_effort,
        "analysis_mode": analysis_mode,
        "auto_sentence_deep_dive": auto_sentence_deep_dive,
        "auto_translation_guidance": auto_translation_guidance,
        "auto_notion_sync": auto_notion_sync,
        "notion_configured": notion_state["configured"],
        "billing_tier": billing_tier,
        "budget_jpy": budget_jpy,
        "spent_before_jpy": spent_before_jpy,
        "usd_to_jpy": usd_to_jpy,
    }
