"""OCR and Document Analysis page view component."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys as _sys
import threading
import streamlit as st

from config import (
    GEMINI_MODEL_VISION,
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_MB,
    SUPPORTED_UPLOAD_FORMATS,
)
from modules.cost_estimator import budget_status, estimate_cost, format_cost, sum_costs
from modules.doc_exporter import export_to_docx
from modules.job_store import create_job
from modules.job_workflow import items_source_hash
from modules.multi_image_workflow import combined_text, move_image_item
from modules.ocr_engine import run_ocr
from modules.result_exporter import (
    analysis_json_bytes,
    default_export_stem,
    markdown_bytes,
    safe_export_stem,
)

from components.helpers import (
    display_rows,
    render_grammar_points,
    render_important_vocabulary,
)


def render_ocr_tab(
    config: dict,
    add_sources_fn,
    remove_image_fn,
    clear_analysis_fn,
    analysis_pages_fn,
    run_item_ocr_fn,
    persist_items_fn,
    persist_analysis_fn,
    text_analyzer_module,
    worker_path: str,
    project_dir: str,
) -> None:
    """Render Tab 1: Image/PDF OCR & AI Text Analysis."""
    items = st.session_state.image_items
    show_preprocessing = config["show_preprocessing"]
    ocr_model_choice = config["ocr_model_choice"]
    text_model_choice = config["text_model_choice"]
    analysis_language = config["analysis_language"]
    billing_tier = config["billing_tier"]
    usd_to_jpy = config["usd_to_jpy"]

    # ── Upload Section ──
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

            if uploaded_files and st.button("➕ Thêm file đã chọn", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý file upload..."):
                    added_any = add_sources_fn([(file.name, file.getvalue()) for file in uploaded_files])
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
            if camera_file and st.button("➕ Thêm ảnh vừa chụp", use_container_width=True):
                if add_sources_fn([(f"camera_{len(items) + 1}.jpg", camera_file.getvalue())]):
                    st.session_state.camera_version += 1
                    st.rerun()

    if not items:
        st.info("Hãy thêm một hoặc nhiều ảnh/PDF để bắt đầu.")
        return

    # ── Image Batch Controls ──
    st.subheader(f"Ảnh/trang PDF trong bộ phân tích ({len(items)})")
    controls_left, controls_middle, controls_right = st.columns(3)

    with controls_left:
        if st.button("🔍 OCR tất cả ảnh chưa xử lý", type="primary", use_container_width=True):
            pending = [item for item in items if not item["ocr_result"]]
            if not pending:
                st.info("Tất cả ảnh/trang đã có OCR.")
            else:
                progress = st.progress(0, text="Đang OCR...")
                done_count = 0

                def _ocr_work(item_to_ocr: dict) -> dict:
                    item_to_ocr["ocr_error"] = None
                    try:
                        result = run_ocr(
                            item_to_ocr["processed_image_bytes"],
                            item_to_ocr["report"],
                            model_name=ocr_model_choice,
                        )
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
                persist_items_fn()
                progress.empty()
                clear_analysis_fn()
                st.rerun()

    with controls_middle:
        if st.button("🔁 OCR/OCR lại toàn bộ ảnh", use_container_width=True):
            progress = st.progress(0, text="Đang OCR toàn bộ...")
            done_count = 0

            def _ocr_work_all(item_to_ocr: dict) -> dict:
                item_to_ocr["ocr_error"] = None
                try:
                    result = run_ocr(
                        item_to_ocr["processed_image_bytes"],
                        item_to_ocr["report"],
                        model_name=ocr_model_choice,
                    )
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
            persist_items_fn()
            progress.empty()
            clear_analysis_fn()
            st.rerun()

    with controls_right:
        ready_count = sum(bool(item.get("edited_text", "").strip()) for item in items)
        st.info(f"Sẵn sàng phân tích: {ready_count}/{len(items)} ảnh")

    # ── Image List Expanders ──
    for index, item in enumerate(items, 1):
        status = "✅ Đã OCR" if item["ocr_result"] else "⏳ Chưa OCR"
        with st.expander(f"Ảnh {index}: {item['name']} · {status}", expanded=len(items) == 1):
            original_col, processed_col = st.columns(2)
            with original_col:
                st.caption("Ảnh gốc")
                st.image(item["original_image_bytes"], use_container_width=True)
            with processed_col:
                st.caption("Ảnh đã xử lý")
                st.image(item["processed_image_bytes"], use_container_width=True)

            report = item["report"]
            if show_preprocessing:
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Chất lượng", report["quality_level"])
                metric2.metric("Góc xoay", f"{report['rotation_detected']}°")
                metric3.metric("Blur score", f"{report['blur_score']:.1f}")

            action1, action2, action3, action4 = st.columns(4)
            if action1.button("🔍 OCR ảnh này", key=f"ocr_{item['id']}", use_container_width=True):
                with st.spinner(f"Đang OCR {item['name']}..."):
                    run_item_ocr_fn(item, model_name=ocr_model_choice)
                persist_items_fn()
                clear_analysis_fn()
                st.rerun()

            if action2.button("⬆️ Lên", key=f"up_{item['id']}", disabled=index == 1, use_container_width=True):
                st.session_state.image_items = move_image_item(items, item["id"], -1)
                persist_items_fn()
                clear_analysis_fn()
                st.rerun()

            if action3.button("⬇️ Xuống", key=f"down_{item['id']}", disabled=index == len(items), use_container_width=True):
                st.session_state.image_items = move_image_item(items, item["id"], 1)
                persist_items_fn()
                clear_analysis_fn()
                st.rerun()

            if action4.button("🗑️ Xóa", key=f"remove_{item['id']}", use_container_width=True):
                remove_image_fn(item["id"])
                st.rerun()

            if item["ocr_error"]:
                st.error(f"❌ Lỗi OCR: {item['ocr_error']}")
            if item["ocr_result"]:
                result = item["ocr_result"]
                meta1, meta2, meta3 = st.columns(3)
                meta1.metric("Hướng chữ", result["text_direction"])
                meta2.metric("Furigana", "Có" if result["has_furigana"] else "Không")
                meta3.metric("Độ tin cậy", result["confidence"])

                edited_text = st.text_area(
                    "Văn bản ảnh này (có thể chỉnh sửa):",
                    value=item["edited_text"],
                    height=180,
                    key=f"text_{item['id']}",
                )
                if edited_text != item["edited_text"]:
                    item["edited_text"] = edited_text
                    persist_items_fn()
                    clear_analysis_fn()
                ocr_result_model = result.get("model_used") or ocr_model_choice
                ocr_cost = estimate_cost(result.get("usage"), ocr_result_model, billing_tier)
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

    # ── Page-by-Page Analysis Section ──
    st.subheader("🧠 Phân tích theo từng trang")
    analysis_text = combined_text(items)
    pages_to_analyze = analysis_pages_fn(items)
    partial = st.session_state.partial_page_analyses

    reasoning_effort = config.get("reasoning_effort", "standard")

    if not analysis_text:
        st.warning("Chưa có văn bản OCR. Hãy OCR ít nhất một ảnh trước khi phân tích.")
    else:
        with st.expander("Xem văn bản OCR theo thứ tự trang"):
            st.text_area("Nội dung gộp theo thứ tự ảnh", value=analysis_text, height=260, disabled=True)
            st.caption(
                "Khi bấm phân tích, app sẽ gọi Gemini riêng cho từng trang/ảnh rồi mới tổng hợp. "
                "Cách này tránh việc file nhiều trang bị dồn quá tải và chỉ phân tích trang đầu."
            )

        # Automatic recovery for interrupted runs
        done_page_indices = {p["page_index"] for p in partial}
        remaining_pages = [p for p in pages_to_analyze if p["page_index"] not in done_page_indices]

        active_background_job = bool(st.query_params.get("job_id"))
        if partial and remaining_pages and active_background_job:
            st.info(
                f"Đã lưu kết quả {len(partial)}/{len(pages_to_analyze)} trang. "
                "Worker nền vẫn đang xử lý các trang còn lại."
            )
        elif partial and remaining_pages:
            st.info(
                f"🔄 Phát hiện phân tích trước đó bị gián đoạn. Đang tự động chạy tiếp {len(remaining_pages)} trang còn lại..."
            )
            if st.button("🔁 Hủy và Phân tích lại từ đầu", use_container_width=True):
                st.session_state.partial_page_analyses = []
                persist_analysis_fn()
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

                resumed_pages = list(partial)

                def _resume_page_done(page_result: dict) -> None:
                    resumed_pages.append(page_result)
                    resumed_pages.sort(key=lambda page: int(page.get("page_index", 0)))
                    st.session_state.partial_page_analyses = list(resumed_pages)
                    persist_analysis_fn()

                new_results = text_analyzer_module.run_page_analyses(
                    remaining_pages,
                    analysis_language=analysis_language,
                    progress_callback=_resume_cb,
                    page_done_callback=_resume_page_done,
                    model_name=text_model_choice,
                    reasoning_effort=reasoning_effort,
                )
                all_page_analyses = partial + new_results.get("page_analyses", [])
                st.session_state.analysis = text_analyzer_module.merge_page_analyses(
                    all_page_analyses, analysis_language=analysis_language
                )
                st.session_state.partial_page_analyses = []
                persist_analysis_fn()
                progress.empty()
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Lỗi tự động phân tích tiếp tục: {exc}")

        if analysis_language == "japanese":
            if st.button("🧠 Phân tích bằng Gemini", type="primary", use_container_width=True):
                clear_analysis_fn()
                detected_lang = "pdf_ja"
                input_data = {
                    "pages": pages_to_analyze,
                    "model_name": text_model_choice,
                    "reasoning_effort": reasoning_effort,
                }
                input_text = json.dumps(input_data)
                job_id = create_job(
                    input_text,
                    detected_lang,
                    session_id=st.session_state.session_id,
                    source_hash=items_source_hash(items),
                )
                subprocess.Popen(
                    [_sys.executable, worker_path, job_id],
                    stdout=subprocess.DEVNULL,
                    stderr=open(project_dir + "/worker_error.log", "a"),
                    cwd=project_dir,
                )
                st.session_state.current_job_id = job_id
                st.query_params["job_id"] = job_id
                st.success(f"Đã bắt đầu phân tích! Job ID: {job_id}")
                st.info("Bạn có thể đóng tab này — kết quả sẽ được lưu lại. Mở lại link có job_id để xem kết quả.")
                st.rerun()
        else:
            if st.button("🧠 Phân tích từng trang đã OCR", type="primary", use_container_width=True):
                clear_analysis_fn()
                detected_lang = "pdf_en"
                input_data = {
                    "pages": pages_to_analyze,
                    "model_name": text_model_choice,
                    "reasoning_effort": reasoning_effort,
                }
                input_text = json.dumps(input_data)
                job_id = create_job(
                    input_text,
                    detected_lang,
                    session_id=st.session_state.session_id,
                    source_hash=items_source_hash(items),
                )
                subprocess.Popen(
                    [_sys.executable, worker_path, job_id],
                    stdout=subprocess.DEVNULL,
                    stderr=open(project_dir + "/worker_error.log", "a"),
                    cwd=project_dir,
                )
                st.session_state.current_job_id = job_id
                st.query_params["job_id"] = job_id
                st.success(f"Đã bắt đầu phân tích! Job ID: {job_id}")
                st.info("Bạn có thể đóng tab này — kết quả sẽ được lưu lại. Mở lại link có job_id để xem kết quả.")
                st.rerun()

    # ── Analysis Report Display ──
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        used_model_name = analysis.get("model_used") or text_model_choice
        st.caption(f"🤖 **Model đã phân tích:** `{used_model_name}` | **Model Vision (OCR):** `{ocr_model_choice}`")
        result_language = analysis.get("analysis_language", analysis_language)

        ocr_costs = [
            estimate_cost(
                item["ocr_result"].get("usage"),
                item["ocr_result"].get("model_used") or ocr_model_choice,
                billing_tier,
            )
            for item in items
            if item.get("ocr_result")
        ]
        analysis_cost = estimate_cost(analysis.get("usage"), used_model_name, billing_tier)
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

            budget = budget_status(
                config.get("budget_jpy", 0),
                config.get("spent_before_jpy", 0),
                float(session_cost["total_cost_usd"]),
                usd_to_jpy,
            )
            if budget["budget_jpy"] > 0:
                budget1, budget2, budget3 = st.columns(3)
                budget1.metric("Ngân sách", f"¥{budget['budget_jpy']:,.0f}")
                budget2.metric("Đã dùng ước tính", f"¥{budget['total_spent_jpy']:,.0f}")
                budget3.metric("Còn lại", f"¥{budget['remaining_jpy']:,.0f}")

        with st.expander("💾 Lưu kết quả phân tích", expanded=True):
            st.caption("Tải file về máy/điện thoại để xem lại sau. JSON lưu dữ liệu có cấu trúc, không nhúng ảnh gốc.")
            export_stem = safe_export_stem(
                st.text_input("Tên file lưu:", value=default_export_stem(items), key="analysis_export_stem")
            )
            docx_name = f"{export_stem}.docx"
            md_name = f"{export_stem}.md"
            json_name = f"{export_stem}.json"
            docx_bytes = export_to_docx(analysis["full_markdown"], docx_name)
            json_bytes = analysis_json_bytes(
                items, analysis, session_cost, billing_tier, usd_to_jpy, budget=budget
            )

            save_col1, save_col2, save_col3 = st.columns(3)
            save_col1.download_button(
                "⬇️ Word .docx",
                data=docx_bytes,
                file_name=docx_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
            save_col2.download_button(
                "⬇️ Markdown .md",
                data=markdown_bytes(analysis),
                file_name=md_name,
                mime="text/markdown",
                use_container_width=True,
            )
            save_col3.download_button(
                "⬇️ Dữ liệu .json",
                data=json_bytes,
                file_name=json_name,
                mime="application/json",
                use_container_width=True,
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
                    st.dataframe(display_rows(page.get("vocabulary_all", []), result_language), use_container_width=True)
                render_important_vocabulary(page.get("vocabulary_important", []))

                if result_language == "japanese":
                    st.subheader("漢字 Kanji")
                    if page.get("kanji_analysis"):
                        st.dataframe(display_rows(page.get("kanji_analysis", []), result_language), use_container_width=True)
                    else:
                        st.info("Trang này chưa có dữ liệu Kanji riêng.")
                else:
                    st.subheader("🔗 Cụm từ & thành ngữ")
                    if page.get("phrasal_collocations"):
                        st.dataframe(display_rows(page.get("phrasal_collocations", []), result_language), use_container_width=True)
                    else:
                        st.info("Trang này chưa có cụm động từ/collocation riêng.")

                st.subheader("🔗 Từ nối" if result_language == "japanese" else "🔗 Từ nối & dấu hiệu diễn ngôn")
                page_marker_key = "connectors" if result_language == "japanese" else "discourse_markers"
                if page.get(page_marker_key):
                    st.dataframe(display_rows(page.get(page_marker_key, []), result_language), use_container_width=True)
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
                            if pattern.get("components"):
                                st.markdown(f"**Thành phần câu:** {pattern['components']}")
                            if pattern.get("function"):
                                st.markdown(f"**Chức năng giao tiếp:** {pattern['function']}")
                            if pattern.get("explanation"):
                                st.markdown(f"**Giải thích:** {pattern['explanation']}")
                else:
                    st.info("Trang này chưa có mẫu câu riêng.")

                with st.expander("💾 Xem Markdown đầy đủ của trang"):
                    st.markdown(page.get("full_markdown") or "Không có dữ liệu.")

        # ── AI Quality Review expander ──
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
