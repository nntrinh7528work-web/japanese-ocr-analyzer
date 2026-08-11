"""YouTube and uploaded-video learning workspace."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import uuid

import streamlit as st

from modules import session_store
from modules.cost_estimator import estimate_run_costs
from modules.doc_exporter import export_to_docx
from modules.job_store import create_job
from modules.video_analyzer import (
    build_cost_estimate,
    build_segments,
    clean_transcript,
    format_timestamp,
    parse_youtube_url,
    probe_video_duration,
    transcript_hash,
    validate_video_upload,
)


def _range_inputs(source: dict, key_prefix: str) -> tuple[float, float]:
    duration = max(1.0, float(source.get("duration_seconds") or 3600))
    saved = source.get("metadata") or {}
    start = st.number_input(
        "Bắt đầu (giây)", min_value=0.0, max_value=max(0.0, duration - 1),
        value=min(float(saved.get("range_start") or 0), max(0.0, duration - 1)), step=30.0,
        key=f"{key_prefix}_start",
    )
    default_end = min(duration, start + 3600)
    end = st.number_input(
        "Kết thúc (giây)", min_value=start + 1, max_value=duration,
        value=min(duration, max(start + 1, float(saved.get("range_end") or default_end))), step=30.0,
        key=f"{key_prefix}_end",
    )
    if end - start > 3600:
        st.warning("Khoảng đã chọn vượt 60 phút.")
    return float(start), float(end)


def _apply_caption_range(source: dict, start: float, end: float, config: dict) -> None:
    if end <= start or end - start > 3600:
        raise ValueError("Khoảng video phải lớn hơn 0 và không vượt 60 phút.")
    selected = [
        row for row in source.get("raw_transcript") or []
        if float(row.get("end", 0) or 0) >= start and float(row.get("start", 0) or 0) <= end
    ]
    if not selected:
        raise ValueError("Khoảng đã chọn không có transcript.")
    clean_rows, cleanup_warnings = clean_transcript(selected)
    selected_hash = transcript_hash(selected)
    metadata = {**(source.get("metadata") or {}), "range_start": start, "range_end": end}
    session_store.update_video_source(
        source["source_id"], metadata=metadata, clean_transcript=clean_rows,
        transcript_warnings=[*(source.get("transcript_warnings") or []), *cleanup_warnings],
        transcript_hash=selected_hash, status="segmenting", error="",
    )
    session_store.update_document_source_hash(
        source["document_id"], selected_hash, status="awaiting_cost_confirmation"
    )
    segments = build_segments(clean_rows)
    session_store.replace_video_segments(source["source_id"], segments)
    refreshed = session_store.get_video_source(source["source_id"]) or source
    estimate = build_cost_estimate(
        refreshed, session_store.list_video_segments(source["source_id"]), config.get("billing_tier", "free")
    )
    session_store.update_video_source(
        source["source_id"], cost_estimate=estimate, status="awaiting_cost_confirmation"
    )


def _start_worker(worker_path: str, project_dir: str, job_id: str) -> None:
    subprocess.Popen(
        [sys.executable, worker_path, job_id],
        stdout=subprocess.DEVNULL,
        stderr=open(str(Path(project_dir) / "worker_error.log"), "a"),
        cwd=project_dir,
    )
    st.query_params["job_id"] = job_id


def _queue_ingest(
    source: dict, config: dict, worker_path: str, project_dir: str, *, allow_gemini: bool,
) -> None:
    payload = {
        "allow_gemini": allow_gemini,
        "billing_tier": config.get("billing_tier", "free"),
        "preferred_language": (session_store.get_document(source["document_id"]) or {}).get("language", "unknown"),
    }
    job_id = create_job(
        json.dumps(payload, ensure_ascii=False), "video", st.session_state.session_id,
        document_id=source["document_id"], job_kind="video_ingest",
        source_id=source["source_id"], stage="pending",
    )
    session_store.update_video_source(source["source_id"], status="ingesting", error="")
    _start_worker(worker_path, project_dir, job_id)


def _create_video_document(title: str) -> dict:
    document = session_store.create_document(
        st.session_state.session_id, title, language="unknown", language_source="video", document_type="video"
    )
    st.session_state.active_document_id = document["document_id"]
    st.session_state.loaded_document_id = None
    st.query_params["document"] = document["document_id"]
    return document


def _render_new_source(config: dict, worker_path: str, project_dir: str) -> None:
    st.subheader("Tạo bài học từ YouTube hoặc file video")
    st.caption("Mỗi video được tạo thành một bài độc lập. App không trộn transcript với bài ảnh/PDF đang mở.")
    url_tab, upload_tab = st.tabs(["Link YouTube", "File video"])
    with url_tab:
        with st.form("video_url_form", clear_on_submit=True):
            value = st.text_input("Link video YouTube công khai", placeholder="https://www.youtube.com/watch?v=...")
            submitted = st.form_submit_button("Lấy caption và lập mục lục", use_container_width=True)
        if submitted:
            try:
                parsed = parse_youtube_url(value)
                document = _create_video_document(f"YouTube {parsed['video_id']}")
                source = session_store.create_video_source(
                    document["document_id"], "youtube", source_url=parsed["canonical_url"],
                    video_id=parsed["video_id"], metadata={"title": f"YouTube {parsed['video_id']}"},
                    status="pending",
                )
                _queue_ingest(source, config, worker_path, project_dir, allow_gemini=False)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with upload_tab:
        with st.form("video_upload_form", clear_on_submit=True):
            uploaded = st.file_uploader(
                "MP4, MOV, WEBM, MPEG hoặc AVI (tối đa 100 MB, 60 phút)",
                type=["mp4", "mov", "webm", "mpeg", "avi"],
            )
            submitted = st.form_submit_button("Tạo bài video", use_container_width=True)
        if submitted:
            if uploaded is None:
                st.warning("Hãy chọn một file video.")
            else:
                try:
                    data = uploaded.getvalue()
                    checked = validate_video_upload(uploaded.name, data, uploaded.type)
                    upload_dir = Path(project_dir) / "data" / "video_uploads" / st.session_state.session_id
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    local_path = upload_dir / f"{uuid.uuid4().hex}.{checked['suffix']}"
                    local_path.write_bytes(data)
                    try:
                        duration = probe_video_duration(str(local_path))
                    except Exception:
                        local_path.unlink(missing_ok=True)
                        raise
                    document = _create_video_document(Path(uploaded.name).stem)
                    session_store.create_video_source(
                        document["document_id"], "upload", file_name=uploaded.name,
                        mime_type=checked["mime_type"], local_path=str(local_path),
                        duration_seconds=duration, metadata={"title": Path(uploaded.name).stem},
                        status="awaiting_ingest_confirmation",
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _render_cost_confirmation(
    source: dict, segments: list[dict], config: dict, worker_path: str, project_dir: str,
) -> None:
    estimate = source.get("cost_estimate") or build_cost_estimate(
        source, segments, config.get("billing_tier", "free")
    )
    rate = float(config.get("usd_to_jpy", 155.0) or 155.0)
    ingest = estimate.get("ingest") or {}
    expected = estimate.get("analysis_expected") or estimate.get("expected") or {}
    maximum = estimate.get("analysis_maximum") or estimate.get("maximum") or {}
    cols = st.columns(4)
    cols[0].metric("Đã dùng lấy transcript", f"¥{float(ingest.get('paid_equivalent_usd', 0)) * rate:,.2f}")
    cols[1].metric("Dự kiến phân tích", f"¥{float(expected.get('paid_equivalent_usd', 0)) * rate:,.2f}")
    cols[2].metric("Tối đa phân tích", f"¥{float(maximum.get('paid_equivalent_usd', 0)) * rate:,.2f}")
    cols[3].metric("Số đoạn / batch", f"{len(segments)} / {estimate.get('batch_count', 0)}")
    st.caption(
        f"Dự kiến {estimate.get('hard_sentence_count', 0)} câu khó. Bảng giá phiên bản "
        f"{estimate.get('pricing_effective_date', 'không rõ')}; free tier vẫn hiện mức tương đương trả phí."
    )
    analyze_hard = st.checkbox("Giải mã tối đa một câu khó mỗi đoạn (tối đa 15 câu)", value=True)
    if st.button("Xác nhận và phân tích toàn bộ video", type="primary", use_container_width=True):
        snapshot = [
            {
                "id": row["segment_id"], "name": row.get("title"),
                "edited_text": row.get("clean_text", ""), "detected_language": row.get("language", "unknown"),
            }
            for row in segments
        ]
        source_hash = str(source.get("transcript_hash") or "")
        version = session_store.create_analysis_version(
            source["document_id"], source_hash, "mixed", "video_balanced",
            config.get("text_model_choice"), status="running", source_items=snapshot,
        )
        payload = {
            "deep_model": config.get("text_model_choice"),
            "reasoning_effort": config.get("reasoning_effort", "standard"),
            "analyze_hard_sentences": analyze_hard,
            "billing_tier": config.get("billing_tier", "free"),
        }
        job_id = create_job(
            json.dumps(payload, ensure_ascii=False), "video", st.session_state.session_id,
            source_hash=source_hash, document_id=source["document_id"], version_id=version["version_id"],
            job_kind="video_analysis", source_id=source["source_id"], stage="pending",
        )
        session_store.save_analysis_version(version["version_id"], status="running", job_id=job_id)
        session_store.update_video_source(source["source_id"], status="analyzing", error="")
        st.session_state.working_version_id = version["version_id"]
        _start_worker(worker_path, project_dir, job_id)
        st.rerun()


def _render_segment(
    segment: dict, source: dict, worker_path: str, project_dir: str, editable: bool = True,
) -> None:
    analysis = segment.get("analysis") or {}
    status = {"done": "Đã xong", "failed": "Lỗi", "running": "Đang xử lý"}.get(segment.get("status"), "Chờ")
    label = (
        f"{format_timestamp(segment.get('start_seconds', 0))} - {format_timestamp(segment.get('end_seconds', 0))} "
        f"| {analysis.get('title') or segment.get('title')} | {status}"
    )
    with st.expander(label, expanded=False):
        if source.get("source_url"):
            timestamp_url = f"{source['source_url']}&t={int(segment.get('start_seconds', 0))}s"
            st.markdown(f"[Mở YouTube tại {format_timestamp(segment.get('start_seconds', 0))}]({timestamp_url})")
        key = f"video_segment_text_{segment['segment_id']}"
        text = st.text_area("Transcript sạch", value=segment.get("clean_text") or "", key=key, height=130)
        if editable and text != (segment.get("clean_text") or ""):
            if st.button("Lưu transcript đã chỉnh", key=f"save_{segment['segment_id']}"):
                session_store.update_video_segment(segment["segment_id"], clean_text=text, status="pending")
                st.success("Đã lưu. Đoạn này cần được phân tích lại trong phiên bản mới.")
                st.rerun()
        if segment.get("error"):
            st.warning(segment["error"])
        if not analysis:
            return
        st.markdown(f"**Tóm tắt:** {analysis.get('summary') or 'Chưa có'}")
        st.markdown(f"**Dịch tự nhiên:** {analysis.get('natural_translation') or 'Chưa có'}")
        for point in analysis.get("key_points") or []:
            st.markdown(f"- {point}")
        turns = analysis.get("dialogue_turns") or []
        if turns:
            st.markdown("#### Hội thoại")
            st.dataframe(turns, use_container_width=True, hide_index=True)
        for label_name, key_name in (
            ("Từ vựng", "vocabulary_all"), ("Kanji / cụm từ", "kanji_analysis"),
            ("Từ nối", "connectors"), ("Ngữ pháp", "grammar_points"), ("Mẫu câu", "sentence_patterns"),
        ):
            rows = analysis.get(key_name) or []
            if rows:
                st.markdown(f"#### {label_name}")
                st.dataframe(rows, use_container_width=True, hide_index=True)
        if analysis.get("sentence_breakdown"):
            with st.expander("Giải mã câu dài tám lớp"):
                st.json(analysis["sentence_breakdown"], expanded=False)
        if analysis.get("sentence_analysis_error"):
            st.warning(f"Phần câu dài chưa hoàn tất: {analysis['sentence_analysis_error']}")
        visual = analysis.get("visual_context_detail") or {}
        if visual:
            st.markdown("#### Bối cảnh hình ảnh")
            st.markdown(str(visual.get("summary") or ""))
            if visual.get("visual_cues"):
                st.dataframe(visual["visual_cues"], use_container_width=True, hide_index=True)
        can_visual = bool(source.get("source_url") or (source.get("ingest_usage") or {}).get("gemini_file_uri"))
        if can_visual and st.button("Bổ sung phân tích hình ảnh", key=f"visual_{segment['segment_id']}"):
            version_id = st.session_state.get("selected_version_id") or (
                session_store.get_document(source["document_id"]) or {}
            ).get("active_version_id")
            payload = {"segment_id": segment["segment_id"]}
            job_id = create_job(
                json.dumps(payload), "video", st.session_state.session_id,
                source_hash=source.get("transcript_hash"), document_id=source["document_id"],
                version_id=version_id, job_kind="video_visual", source_id=source["source_id"], stage="pending",
            )
            _start_worker(worker_path, project_dir, job_id)
            st.rerun()


def _video_json_bytes(source: dict, segments: list[dict], analysis: dict) -> bytes:
    payload = {
        "schema_version": "video-1.0",
        "source": {**source, "local_path": None},
        "transcript_raw": source.get("raw_transcript") or [],
        "transcript_clean": source.get("clean_transcript") or [],
        "segments": segments,
        "analysis": analysis,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _render_results(source: dict, segments: list[dict], analysis: dict, config: dict) -> None:
    st.subheader("Kết quả phân tích video")
    markdown = str(analysis.get("full_markdown") or "")
    actual = estimate_run_costs(
        analysis.get("video_analysis_runs") or [], {},
        str(analysis.get("model_used") or "gemini-2.5-flash-lite"),
        config.get("billing_tier", "free"),
    )
    rate = float(config.get("usd_to_jpy", 155.0) or 155.0)
    metrics = st.columns(3)
    metrics[0].metric("Token vào thực tế", f"{int(actual.get('input_tokens', 0)):,}")
    metrics[1].metric("Token ra thực tế", f"{int(actual.get('output_tokens', 0)):,}")
    metrics[2].metric("Chi phí tương đương", f"¥{float(actual.get('paid_equivalent_usd', 0)) * rate:,.2f}")
    cols = st.columns(3)
    cols[0].download_button(
        "Tải Markdown", markdown.encode("utf-8"), "video_analysis.md", "text/markdown", use_container_width=True
    )
    cols[1].download_button(
        "Tải Word", export_to_docx(markdown, "video_analysis.docx"), "video_analysis.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True,
    )
    cols[2].download_button(
        "Tải JSON đầy đủ", _video_json_bytes(source, segments, analysis), "video_analysis.json",
        "application/json", use_container_width=True,
    )


def render_video_tab(
    *, config: dict, active_document: dict, worker_path: str, project_dir: str,
) -> None:
    """Render source creation, two confirmations, progress, and completed segments."""
    workspace = session_store.get_document_workspace(active_document.get("document_id", "")) or {}
    source = workspace.get("video_source") if active_document.get("document_type") == "video" else None
    segments = workspace.get("video_segments") or []
    if source is None:
        _render_new_source(config, worker_path, project_dir)
        return

    title = (source.get("metadata") or {}).get("title") or source.get("file_name") or "Video"
    st.subheader(title)
    if source.get("source_url"):
        st.video(source["source_url"])
    elif source.get("local_path") and Path(source["local_path"]).exists():
        st.video(source["local_path"])
    st.caption(
        f"Nguồn: {'YouTube' if source.get('source_kind') == 'youtube' else 'File tải lên'} | "
        f"Thời lượng: {format_timestamp(source.get('duration_seconds', 0))} | "
        f"Transcript: {source.get('transcript_provider') or 'chưa có'}"
    )

    status = str(source.get("status") or "pending")
    if status in {"pending", "ingesting", "segmenting"}:
        st.info("Đang lấy transcript và tạo mục lục. Bạn có thể chuyển sang bài khác; job vẫn tiếp tục.")
        return
    if status == "awaiting_range_selection":
        st.warning("Video dài hơn 60 phút. Hãy chọn một khoảng tối đa 60 phút để tạo bài học.")
        range_start, range_end = _range_inputs(source, f"caption_range_{source['source_id']}")
        if st.button("Dùng khoảng đã chọn", type="primary", use_container_width=True):
            try:
                _apply_caption_range(source, range_start, range_end, config)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        return
    if status == "awaiting_ingest_confirmation":
        duration = float(source.get("duration_seconds") or 600)
        provisional = build_cost_estimate(
            {**source, "duration_seconds": duration, "transcript_provider": "gemini_video"}, [],
            config.get("billing_tier", "free"),
        )
        jpy = float((provisional.get("ingest") or {}).get("paid_equivalent_usd", 0)) * float(config.get("usd_to_jpy", 155))
        st.warning(
            f"Không lấy được caption miễn phí. Gemini cần đọc video ở độ phân giải thấp. "
            f"Ước tính bước này khoảng ¥{jpy:,.2f}; video chưa biết thời lượng dùng giả định 10 phút."
        )
        if source.get("error"):
            st.caption(f"Lý do caption không dùng được: {source['error']}")
        if float(source.get("duration_seconds") or 0) > 3600:
            st.caption("File dài hơn 60 phút. Gemini chỉ được yêu cầu trả transcript cho khoảng bạn chọn dưới đây.")
            range_start, range_end = _range_inputs(source, f"upload_range_{source['source_id']}")
            metadata = {**(source.get("metadata") or {}), "range_start": range_start, "range_end": range_end}
            session_store.update_video_source(source["source_id"], metadata=metadata)
        if st.button("Xác nhận dùng Gemini để lấy transcript", type="primary", use_container_width=True):
            metadata = session_store.get_video_source(source["source_id"]).get("metadata") or {}
            if float(source.get("duration_seconds") or 0) > 3600 and (
                float(metadata.get("range_end") or 0) - float(metadata.get("range_start") or 0) > 3600
            ):
                st.error("Khoảng video không được vượt 60 phút.")
                return
            _queue_ingest(source, config, worker_path, project_dir, allow_gemini=True)
            st.rerun()
        return
    if status == "awaiting_cost_confirmation":
        st.success(f"Đã có transcript và {len(segments)} đoạn. Chưa gọi phân tích nội dung toàn bộ.")
        _render_cost_confirmation(source, segments, config, worker_path, project_dir)
    elif status == "analyzing":
        done = sum(1 for row in segments if row.get("status") in {"done", "failed"})
        st.info(f"Đang phân tích: {done}/{len(segments)} đoạn đã xử lý. Kết quả từng đoạn được lưu ngay.")
        if segments:
            st.progress(done / len(segments))
    elif status == "analyzed":
        selected_version = (
            session_store.get_analysis_version(st.session_state.get("selected_version_id"))
            if st.session_state.get("selected_version_id") else workspace.get("active_version")
        ) or {}
        analysis = selected_version.get("analysis") or {}
        if analysis:
            selected_segments = analysis.get("video_segments") or segments
            _render_results(source, selected_segments, analysis, config)
            if selected_segments is not segments:
                segments = selected_segments
    elif status == "failed":
        st.error(source.get("error") or "Phân tích segment chưa hoàn tất.")
        st.caption("Transcript vẫn được giữ nguyên. Bạn có thể chạy lại chỉ bước phân tích, không cần lấy transcript lần nữa.")
        if segments and st.button("Chạy lại phân tích từ transcript hiện có", type="primary", use_container_width=True):
            for segment in segments:
                if segment.get("status") == "failed":
                    session_store.update_video_segment(segment["segment_id"], status="pending", error="")
            session_store.update_video_source(source["source_id"], status="awaiting_cost_confirmation", error="")
            st.rerun()
    elif source.get("error"):
        st.error(source["error"])

    if segments:
        st.markdown("### Mục lục và nội dung từng đoạn")
        for segment in segments:
            _render_segment(segment, source, worker_path, project_dir, editable=status != "analyzing")
    st.divider()
    _render_new_source(config, worker_path, project_dir)
