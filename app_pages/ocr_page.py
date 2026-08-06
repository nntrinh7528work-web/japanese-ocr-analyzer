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
from modules.cost_estimator import budget_status, estimate_cost, estimate_run_costs, format_cost, sum_costs
from modules.doc_exporter import export_to_docx
from modules.job_store import create_job
from modules.job_workflow import items_source_hash
from modules.multi_image_workflow import combined_text, move_image_item
from modules.ocr_engine import run_ocr
from modules import session_store
from modules.notion_sync import enqueue_analysis_sync, notion_connection_state
from modules.result_exporter import (
    analysis_json_bytes,
    default_export_stem,
    markdown_bytes,
    safe_export_stem,
)
from modules.sentence_analyzer import analysis_markdown, split_sentences
from modules.tts_engine import get_audio_cache_key, text_to_speech

from components.helpers import (
    display_rows,
    render_grammar_points,
    render_important_vocabulary,
)


def _render_labeled_items(title: str, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        return
    st.markdown(f"**{title}:**")
    for row in rows:
        values = [str(row.get(field) or "").strip() for field in fields]
        st.markdown("- " + " | ".join(value for value in values if value))


def _render_deep_details(row: dict, language: str) -> None:
    st.markdown("#### Giải mã câu dài (8 lớp)")
    segments = row.get("segments") or []
    if segments:
        st.markdown("**Cụm từ có nhãn vai trò**")
        st.dataframe(
            [
                {
                    "Cụm": item.get("text", ""),
                    "Hiragana": item.get("reading", "") if language == "japanese" else "",
                    "Nhãn/Vai trò": item.get("role", ""),
                    "Nghĩa tiếng Việt": item.get("meaning_vi", ""),
                    "Bổ nghĩa cho": item.get("modifies", ""),
                }
                for item in segments
            ],
            use_container_width=True,
            hide_index=True,
        )
    clauses = row.get("clauses") or []
    if clauses:
        st.markdown("**Ranh giới và quan hệ mệnh đề**")
        st.dataframe(
            [
                {
                    "Nhãn mệnh đề": item.get("label", ""),
                    "Nội dung": item.get("text", ""),
                    "Vai trò": item.get("role", ""),
                    "Quan hệ với mệnh đề chính": item.get("relation_to_main", ""),
                }
                for item in clauses
            ],
            use_container_width=True,
            hide_index=True,
        )
    if row.get("structure_summary"):
        st.markdown(f"**Cấu trúc câu:** {row['structure_summary']}")
    _render_labeled_items("Thành phần lược bỏ", row.get("omitted_elements") or [], ("element", "recovered", "reason"))
    _render_labeled_items("Từ quy chiếu", row.get("references") or [], ("expression", "referent", "reason"))
    _render_labeled_items("Luồng logic", row.get("logic") or [], ("marker", "relation", "scope"))
    st.markdown(f"**Câu viết lại đơn giản:** {row.get('simplified_source') or 'Chưa có'}")
    st.markdown(f"**Nghĩa tiếng Việt:** {row.get('simplified_vi') or 'Chưa có'}")
    questions = row.get("questions") or []
    if questions:
        st.markdown("**Câu hỏi kiểm tra hiểu**")
        for question_index, question in enumerate(questions, 1):
            st.write(f"{question_index}. {question.get('question') or ''}")
        show_answers = st.toggle(
            "Hiện đáp án",
            value=False,
            key=f"answers_{row.get('sentence_id')}",
        )
        if show_answers:
            for question_index, question in enumerate(questions, 1):
                st.info(
                    f"{question_index}. {question.get('answer') or 'Chưa có đáp án'}\n\n"
                    f"{question.get('explanation') or ''}"
                )


def _render_audio_control(text: str, lang: str, slow: bool, label: str, key: str) -> None:
    cache_key = "teacher_tts_" + get_audio_cache_key(text, lang, slow)
    if st.button(label, key=key, use_container_width=True, disabled=not bool(text.strip())):
        with st.spinner("Đang tạo audio..."):
            audio = text_to_speech(text, lang=lang, slow=slow)
        if audio:
            st.session_state[cache_key] = audio
        else:
            st.warning("Không thể tạo audio. Kết quả dịch vẫn được giữ nguyên.")
    if st.session_state.get(cache_key):
        st.audio(st.session_state[cache_key], format="audio/mp3")


def _start_guidance_job(
    missing: list[dict],
    page: dict,
    source_text: str,
    language: str,
    model_name: str,
    reasoning_effort: str,
    items: list[dict],
    worker_path: str,
    project_dir: str,
) -> None:
    payload = {
        "catalog": missing,
        "page_text": source_text,
        "page_index": int(page.get("page_index", 0)),
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
    }
    lang = "guidance_ja" if language == "japanese" else "guidance_en"
    job_id = create_job(
        json.dumps(payload, ensure_ascii=False),
        lang,
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
    st.rerun()


def _render_translation_guidance(
    page: dict,
    language: str,
    source_text: str,
    model_name: str,
    reasoning_effort: str,
    items: list[dict],
    persist_analysis_fn,
    worker_path: str,
    project_dir: str,
) -> None:
    st.subheader("Đối chiếu OCR và giáo viên hướng dẫn dịch")
    if not page.get("sentence_catalog"):
        page["sentence_catalog"] = split_sentences(source_text, language, int(page.get("page_index", 1)))
        page.setdefault("sentence_breakdowns", [])
        page.setdefault("sentence_analysis_usage", {})
        persist_analysis_fn()

    if page.get("translation_guidance_errors"):
        st.warning(
            f"Có {len(page['translation_guidance_errors'])} batch hướng dẫn chưa hoàn tất. "
            "Kết quả chính và các câu đã dịch vẫn được giữ."
        )
    if page.get("sentence_analysis_error"):
        st.warning(f"Phân tích chính vẫn được giữ. Giải mã câu dài gặp lỗi: {page['sentence_analysis_error']}")

    guidance = {row.get("sentence_id"): row for row in page.get("translation_guidance", [])}
    breakdowns = {row.get("sentence_id"): row for row in page.get("sentence_breakdowns", [])}
    catalog = sorted(page.get("sentence_catalog") or [], key=lambda row: int(row.get("ordinal", 0)))
    slow_source = st.toggle(
        "Nói chậm nguyên văn",
        value=False,
        key=f"teacher_tts_slow_{page.get('page_index')}",
    )
    source_lang = "ja" if language == "japanese" else "en"

    for sentence in catalog:
        sentence_id = sentence.get("sentence_id")
        row = guidance.get(sentence_id) or {}
        deep = breakdowns.get(sentence_id)
        translations = row.get("translations") or (deep or {}).get("translations") or {}
        original = str(sentence.get("original") or "")
        reading = row.get("reading") or (deep or {}).get("reading") or ""
        labels = [f"Câu {sentence.get('ordinal', '?')}"]
        if sentence.get("eligible"):
            labels.append(f"Câu khó · điểm {sentence.get('complexity_score', 0)}")
        if deep:
            labels.append("Đã giải mã sâu")
        if row.get("ocr_warning"):
            labels.append("Có cảnh báo OCR")
        with st.container(border=True):
            st.markdown("**" + " | ".join(labels) + "**")
            source_col, teacher_col = st.columns(2)
            with source_col:
                st.markdown("**Văn bản OCR đã duyệt**")
                st.code(original, language=None)
                if language == "japanese" and reading:
                    st.caption(f"Hiragana: {reading}")
                _render_audio_control(
                    original,
                    source_lang,
                    slow_source,
                    "Nghe nguyên văn",
                    f"listen_source_{sentence_id}",
                )
            with teacher_col:
                st.markdown("**Giáo viên dịch tự nhiên**")
                natural = str(translations.get("natural") or "")
                if natural:
                    st.success(natural)
                else:
                    st.info("Câu này chưa có hướng dẫn dịch.")
                for point in (row.get("key_points") or [])[:3]:
                    source = f" · `{point.get('source')}`" if point.get("source") else ""
                    st.markdown(
                        f"- **{point.get('label') or 'Điểm mấu chốt'}**{source}: "
                        f"{point.get('explanation_vi') or ''}"
                    )
                _render_audio_control(
                    natural,
                    "vi",
                    False,
                    "Nghe bản dịch tiếng Việt",
                    f"listen_vi_{sentence_id}",
                )

            if row or deep:
                with st.expander(f"Chi tiết dịch và phân tích câu {sentence.get('ordinal', '?')}", expanded=False):
                    st.markdown(f"**Dịch theo cụm:** {translations.get('chunked') or 'Chưa có'}")
                    st.markdown(f"**Dịch sát toàn câu:** {translations.get('literal') or 'Chưa có'}")
                    if row.get("translation_steps"):
                        st.markdown("**Thứ tự dịch đề xuất**")
                        for index, step in enumerate(row["translation_steps"], 1):
                            order = step.get("order") or index
                            st.markdown(
                                f"{order}. `{step.get('source_chunk')}` → "
                                f"{step.get('meaning_vi')}: {step.get('advice_vi')}"
                            )
                    if row.get("ocr_warning"):
                        st.warning(f"Đề xuất kiểm tra OCR: {row['ocr_warning']}")
                    if row.get("related_analysis"):
                        st.markdown("**Phân tích liên quan ở các mục bên dưới**")
                        for ref in row["related_analysis"]:
                            st.markdown(
                                f"- [{ref.get('category_label')}] **{ref.get('label')}**: "
                                f"{ref.get('summary') or 'Xem bảng đầy đủ bên dưới.'}"
                            )
                    if deep:
                        _render_deep_details(deep, language)

    analyzed_ids = set(breakdowns)
    remaining = [row for row in catalog if row.get("sentence_id") not in analyzed_ids]
    missing_guidance = [row for row in catalog if row.get("sentence_id") not in guidance]
    if missing_guidance and st.button(
        f"Tạo hướng dẫn cho {len(missing_guidance)} câu còn thiếu",
        key=f"run_guidance_{page.get('page_index')}",
        type="primary",
        use_container_width=True,
    ):
        _start_guidance_job(
            missing_guidance,
            page,
            source_text,
            language,
            model_name,
            reasoning_effort,
            items,
            worker_path,
            project_dir,
        )

    with st.expander("Phân tích thêm câu khác", expanded=False):
        if not remaining:
            st.info("Tất cả câu trong trang đã được phân tích.")
            return
        options = {f"Câu {row['ordinal']}: {row['original'][:120]}": row for row in remaining}
        selected_label = st.selectbox(
            "Chọn câu chưa xử lý",
            options=list(options),
            key=f"manual_sentence_{page.get('page_index')}",
        )
        selected = options[selected_label]
        st.caption(
            f"Điểm phức tạp: {selected.get('complexity_score', 0)} · "
            + (", ".join(selected.get("complexity_signals") or []) or "câu ngắn/ít cấu trúc lồng")
        )
        if st.button(
            "Phân tích câu đã chọn trong nền",
            key=f"run_manual_sentence_{page.get('page_index')}",
            use_container_width=True,
        ):
            payload = {
                "sentence": selected,
                "page_text": source_text,
                "model_name": model_name,
                "reasoning_effort": reasoning_effort,
            }
            lang = "sentence_ja" if language == "japanese" else "sentence_en"
            job_id = create_job(
                json.dumps(payload, ensure_ascii=False),
                lang,
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
            st.success("Đã đưa câu vào hàng đợi. Có thể đóng tab và mở lại đúng link phiên này.")
            st.rerun()


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
    notion_worker_path: str,
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
            pending = [item for item in items if not str(item.get("edited_text") or "").strip()]
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
    missing_ocr_items = [
        (index, item.get("name") or f"Trang {index}")
        for index, item in enumerate(items, 1)
        if not str(item.get("edited_text") or "").strip()
    ]
    all_pages_ready = not missing_ocr_items and len(pages_to_analyze) == len(items)
    partial = st.session_state.partial_page_analyses

    reasoning_effort = config.get("reasoning_effort", "standard")

    if not analysis_text:
        st.warning("Chưa có văn bản OCR. Hãy OCR ít nhất một ảnh trước khi phân tích.")
    else:
        if missing_ocr_items:
            missing_names = ", ".join(f"trang {index} ({name})" for index, name in missing_ocr_items)
            st.error(
                "Chưa thể phân tích toàn bộ tài liệu vì còn trang chưa có OCR: "
                f"{missing_names}. Hãy bấm 'OCR tất cả ảnh chưa xử lý' rồi thử lại."
            )
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
                    page_index = int(page_result.get("page_index", 0))
                    resumed_pages[:] = [
                        page for page in resumed_pages
                        if int(page.get("page_index", 0)) != page_index
                    ]
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
                    auto_sentence_deep_dive=config.get("auto_sentence_deep_dive", True),
                    auto_translation_guidance=config.get("auto_translation_guidance", True),
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
            if st.button(
                "🧠 Phân tích bằng Gemini",
                type="primary",
                use_container_width=True,
                disabled=not all_pages_ready,
            ):
                clear_analysis_fn()
                detected_lang = "pdf_ja"
                input_data = {
                    "pages": pages_to_analyze,
                    "model_name": text_model_choice,
                    "reasoning_effort": reasoning_effort,
                    "auto_sentence_deep_dive": config.get("auto_sentence_deep_dive", True),
                    "auto_translation_guidance": config.get("auto_translation_guidance", True),
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
            if st.button(
                "🧠 Phân tích từng trang đã OCR",
                type="primary",
                use_container_width=True,
                disabled=not all_pages_ready,
            ):
                clear_analysis_fn()
                detected_lang = "pdf_en"
                input_data = {
                    "pages": pages_to_analyze,
                    "model_name": text_model_choice,
                    "reasoning_effort": reasoning_effort,
                    "auto_sentence_deep_dive": config.get("auto_sentence_deep_dive", True),
                    "auto_translation_guidance": config.get("auto_translation_guidance", True),
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
        sentence_cost = estimate_run_costs(
            analysis.get("sentence_analysis_runs"),
            analysis.get("sentence_analysis_usage"),
            analysis.get("sentence_analysis_model") or used_model_name,
            billing_tier,
        )
        guidance_cost = estimate_run_costs(
            analysis.get("translation_guidance_runs"),
            analysis.get("translation_guidance_usage"),
            analysis.get("translation_guidance_model") or used_model_name,
            billing_tier,
        )
        session_cost = sum_costs([*ocr_costs, analysis_cost, guidance_cost, sentence_cost])

        with st.expander("💰 Tổng chi phí phiên phân tích", expanded=True):
            cost1, cost2 = st.columns(2)
            cost1.metric("OCR ảnh", format_cost(sum(float(cost["total_cost_usd"]) for cost in ocr_costs), usd_to_jpy))
            cost2.metric("Phân tích văn bản", format_cost(analysis_cost["total_cost_usd"], usd_to_jpy))
            cost3, cost4 = st.columns(2)
            cost3.metric("Giáo viên hướng dẫn dịch", format_cost(guidance_cost["total_cost_usd"], usd_to_jpy))
            cost4.metric("Giải mã câu dài", format_cost(sentence_cost["total_cost_usd"], usd_to_jpy))
            cost5, cost6 = st.columns(2)
            cost5.metric("Tổng token", f"{session_cost['input_tokens'] + session_cost['output_tokens']:,}")
            cost6.metric("Tổng ước tính", format_cost(session_cost["total_cost_usd"], usd_to_jpy))
            st.caption("Audio Edge TTS/gTTS được tạo khi bấm và không tính vào token Gemini.")

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
            docx_bytes = export_to_docx(analysis_markdown(analysis), docx_name)
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

            st.divider()
            st.markdown("**Lưu tự động vào Notion**")
            st.caption(
                "Mỗi kết quả khác nhau được giữ thành một phiên bản riêng, kèm file JSON gốc "
                "và mã SHA-256. Các cột Notion là bản tóm lược để lọc và ôn tập."
            )
            notion_state = notion_connection_state()
            sync_run = session_store.get_notion_sync_for_source(
                st.session_state.session_id,
                items_source_hash(items),
            )
            if not notion_state["configured"]:
                st.info(
                    "Notion chưa được kết nối. Cấu hình NOTION_TOKEN và "
                    "NOTION_PARENT_PAGE_ID trong Streamlit Secrets để bật đồng bộ."
                )
            elif sync_run:
                status = sync_run.get("status")
                if status == "done":
                    st.success("Đã lưu bài phân tích nguyên trạng và các mục cần học vào Notion.")
                elif status == "partial":
                    st.warning(sync_run.get("error") or "Bài đã lưu nhưng còn mục cần thử lại.")
                elif status in ("pending", "queued", "running"):
                    st.info(
                        f"Đang đồng bộ Notion: {sync_run.get('completed_items', 0)}/"
                        f"{sync_run.get('total_items', 0)} mục."
                    )
                    st.button("🔄 Cập nhật trạng thái Notion", key="refresh_notion_status")
                elif status == "retry":
                    st.warning(
                        "Notion tạm thời chưa nhận dữ liệu. App sẽ tự thử lại khi đến hạn. "
                        + str(sync_run.get("error") or "")
                    )
                else:
                    st.error(f"Đồng bộ Notion thất bại: {sync_run.get('error') or 'Không rõ lỗi'}")

                if sync_run.get("notion_page_url"):
                    st.link_button(
                        "Mở bài trong Notion",
                        sync_run["notion_page_url"],
                        use_container_width=True,
                    )
                if status in ("partial", "retry", "failed") and st.button(
                    "Thử đồng bộ Notion lại",
                    key="retry_notion_sync",
                    use_container_width=True,
                ):
                    session_store.retry_notion_sync_run(sync_run["run_id"])
                    if session_store.dispatch_notion_sync_run(sync_run["run_id"]):
                        subprocess.Popen(
                            [_sys.executable, notion_worker_path, sync_run["run_id"]],
                            stdout=subprocess.DEVNULL,
                            stderr=open(project_dir + "/notion_worker_error.log", "a"),
                            cwd=project_dir,
                        )
                    st.rerun()
            elif st.button("Lưu bài này vào Notion ngay", use_container_width=True):
                run = enqueue_analysis_sync(
                    st.session_state.session_id,
                    items,
                    analysis,
                    billing_tier=billing_tier,
                    usd_to_jpy=usd_to_jpy,
                    force=True,
                )
                if session_store.dispatch_notion_sync_run(run["run_id"]):
                    subprocess.Popen(
                        [_sys.executable, notion_worker_path, run["run_id"]],
                        stdout=subprocess.DEVNULL,
                        stderr=open(project_dir + "/notion_worker_error.log", "a"),
                        cwd=project_dir,
                    )
                st.rerun()

        page_analyses = analysis.get("page_analyses") or [analysis]
        tab_titles = [page.get("source_label") or page.get("page_name") or f"Trang {i+1}" for i, page in enumerate(page_analyses)]

        tabs = st.tabs(tab_titles)
        for index, (tab, page) in enumerate(zip(tabs, page_analyses)):
            with tab:
                source_page = next(
                    (candidate for candidate in pages_to_analyze if int(candidate.get("page_index", 0)) == int(page.get("page_index", index + 1))),
                    {},
                )
                source_text = page.get("source_text") or source_page.get("text") or ""
                st.subheader("Văn bản OCR đã duyệt")
                st.text_area(
                    "Nguyên văn dùng làm nguồn dịch",
                    value=source_text,
                    height=180,
                    disabled=True,
                    key=f"source_ocr_{page.get('page_index', index + 1)}",
                )
                st.subheader("Tóm tắt nội dung")
                st.info(page.get("summary") or "Không có tóm tắt.")

                _render_translation_guidance(
                    page,
                    result_language,
                    source_text,
                    used_model_name,
                    reasoning_effort,
                    items,
                    persist_analysis_fn,
                    worker_path,
                    project_dir,
                )

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
