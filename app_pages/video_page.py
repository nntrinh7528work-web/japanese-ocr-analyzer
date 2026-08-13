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
    build_video_usage_cost_breakdown,
    clean_transcript,
    cues_to_transcript_rows,
    estimate_audio_transcription_cost,
    estimate_cue_translation_cost,
    format_timestamp,
    normalize_video_segment_result,
    parse_manual_youtube_transcript,
    parse_youtube_url,
    probe_video_metadata,
    transcript_hash,
    transcript_rows_to_cues,
    validate_video_upload,
    validate_video_duration,
)
from components.video_player import render_upload_player, render_youtube_player


def _range_inputs(source: dict, key_prefix: str) -> tuple[float, float]:
    duration = max(1.0, float(source.get("duration_seconds") or 1800))
    saved = source.get("metadata") or {}
    start = st.number_input(
        "Bắt đầu (giây)", min_value=0.0, max_value=max(0.0, duration - 1),
        value=min(float(saved.get("range_start") or 0), max(0.0, duration - 1)), step=30.0,
        key=f"{key_prefix}_start",
    )
    default_end = min(duration, start + 1800)
    end = st.number_input(
        "Kết thúc (giây)", min_value=start + 1, max_value=duration,
        value=min(duration, max(start + 1, float(saved.get("range_end") or default_end))), step=30.0,
        key=f"{key_prefix}_end",
    )
    if end - start > 1800:
        st.warning("Khoảng đã chọn vượt 30 phút.")
    return float(start), float(end)


def _apply_caption_range(source: dict, start: float, end: float, config: dict) -> None:
    if end <= start or end - start > 1800:
        raise ValueError("Khoảng video phải lớn hơn 0 và không vượt 30 phút.")
    selected = [
        row for row in source.get("raw_transcript") or []
        if float(row.get("end", 0) or 0) >= start and float(row.get("start", 0) or 0) <= end
    ]
    if not selected:
        raise ValueError("Khoảng đã chọn không có transcript.")
    clean_rows, cleanup_warnings = clean_transcript(selected)
    selected_cues = [
        cue for cue in session_store.list_video_cues(source["source_id"])
        if float(cue.get("end_seconds", 0) or 0) >= start
        and float(cue.get("start_seconds", 0) or 0) <= end
    ]
    if selected_cues:
        session_store.replace_video_cues(source["source_id"], selected_cues)
    selected_hash = transcript_hash(selected)
    metadata = {**(source.get("metadata") or {}), "range_start": start, "range_end": end}
    translated = selected_cues and all(cue.get("translation_vi") for cue in selected_cues)
    next_status = "awaiting_cost_confirmation" if translated else "awaiting_translation_confirmation"
    session_store.update_video_source(
        source["source_id"], metadata=metadata, clean_transcript=clean_rows,
        raw_transcript=selected,
        transcript_warnings=[*(source.get("transcript_warnings") or []), *cleanup_warnings],
        transcript_hash=selected_hash, status=next_status, error="",
    )
    session_store.update_document_source_hash(
        source["document_id"], selected_hash, status=next_status
    )
    segments = build_segments(clean_rows, namespace=source["source_id"])
    session_store.replace_video_segments(source["source_id"], segments)
    refreshed = session_store.get_video_source(source["source_id"]) or source
    estimate = build_cost_estimate(
        refreshed, session_store.list_video_segments(source["source_id"]), config.get("billing_tier", "free")
    )
    session_store.update_video_source(
        source["source_id"], cost_estimate=estimate, status=next_status
    )


def _apply_manual_youtube_transcript(source: dict, value: str, config: dict) -> None:
    """Continue a blocked YouTube lesson using user-supplied captions."""
    rows = parse_manual_youtube_transcript(value)
    clean_rows, warnings = clean_transcript(rows)
    cues = transcript_rows_to_cues(rows, source["source_id"], "manual_transcript")
    session_store.replace_video_cues(source["source_id"], cues)
    segments = build_segments(clean_rows, namespace=source["source_id"])
    session_store.replace_video_segments(source["source_id"], segments)
    source_hash = transcript_hash(rows)
    metadata = {
        **(source.get("metadata") or {}),
        "caption_error_code": "",
        "manual_transcript": True,
    }
    session_store.update_video_source(
        source["source_id"], raw_transcript=rows, clean_transcript=clean_rows,
        transcript_warnings=warnings, transcript_provider="manual_transcript",
        transcript_hash=source_hash, metadata=metadata,
        status="awaiting_translation_confirmation", error="",
    )
    session_store.update_document_source_hash(
        source["document_id"], source_hash, status="awaiting_translation_confirmation"
    )
    refreshed = session_store.get_video_source(source["source_id"]) or source
    estimate = build_cost_estimate(refreshed, segments, config.get("billing_tier", "free"))
    session_store.update_video_source(source["source_id"], cost_estimate=estimate)


def _start_worker(
    worker_path: str, project_dir: str, job_id: str, *, document_id: str | None = None,
) -> None:
    subprocess.Popen(
        [sys.executable, worker_path, job_id],
        stdout=subprocess.DEVNULL,
        stderr=open(str(Path(project_dir) / "worker_error.log"), "a"),
        cwd=project_dir,
    )
    if document_id:
        st.session_state.active_document_id = document_id
        st.session_state.loaded_document_id = None
        # Keep the job and its owning lesson together. Updating these query
        # parameters in one operation prevents a rerun from reopening the
        # previously selected image lesson while the video job continues.
        st.query_params.update({"document": document_id, "job_id": job_id})
    else:
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
    _start_worker(worker_path, project_dir, job_id, document_id=source["document_id"])


def _queue_video_job(
    source: dict, config: dict, worker_path: str, project_dir: str, job_kind: str,
) -> None:
    payload = {"billing_tier": config.get("billing_tier", "free")}
    job_id = create_job(
        json.dumps(payload, ensure_ascii=False), "video", st.session_state.session_id,
        document_id=source["document_id"], job_kind=job_kind,
        source_id=source["source_id"], stage="pending",
    )
    status = "transcribing" if job_kind == "video_transcribe" else "translating"
    session_store.update_video_source(source["source_id"], status=status, error="")
    _start_worker(worker_path, project_dir, job_id, document_id=source["document_id"])


def _queue_cue_recheck(
    source: dict, cue_id: str, revision: int, worker_path: str, project_dir: str,
) -> None:
    payload = {"cue_id": cue_id, "revision": int(revision)}
    job_id = create_job(
        json.dumps(payload, ensure_ascii=False), "video", st.session_state.session_id,
        source_hash=str(source.get("transcript_hash") or ""), document_id=source["document_id"],
        job_kind="video_cue_recheck", source_id=source["source_id"], stage="pending",
    )
    _start_worker(worker_path, project_dir, job_id, document_id=source["document_id"])


def _refresh_transcript_after_cue_change(source: dict, config: dict) -> None:
    cues = session_store.list_video_cues(source["source_id"])
    rows = cues_to_transcript_rows(cues)
    clean_rows, warnings = clean_transcript(rows)
    current_hash = transcript_hash(rows)
    translated = all(str(cue.get("translation_vi") or "").strip() for cue in cues)
    status = "awaiting_cost_confirmation" if translated else "awaiting_translation_confirmation"
    segments = session_store.list_video_segments(source["source_id"])
    if not segments:
        session_store.replace_video_segments(source["source_id"], build_segments(clean_rows, namespace=source["source_id"]))
    else:
        for segment in segments:
            segment_rows = [
                row for row in clean_rows
                if float(row.get("end", 0)) >= float(segment.get("start_seconds", 0))
                and float(row.get("start", 0)) <= float(segment.get("end_seconds", 0))
            ]
            updated_text = " ".join(str(row.get("text") or "") for row in segment_rows).strip()
            if updated_text and updated_text != str(segment.get("clean_text") or ""):
                session_store.update_video_segment(
                    segment["segment_id"], original_text=updated_text,
                    clean_text=updated_text, status="pending", error="",
                )
    metadata = {**(source.get("metadata") or {}), "transcript_pipeline_version": 2, "has_unanalyzed_changes": True}
    session_store.update_video_source(
        source["source_id"], raw_transcript=rows, clean_transcript=clean_rows,
        transcript_hash=current_hash, transcript_warnings=[*(source.get("transcript_warnings") or []), *warnings],
        metadata=metadata, status=status, error="",
    )
    session_store.update_document_source_hash(source["document_id"], current_hash, status=status)
    refreshed = session_store.get_video_source(source["source_id"]) or source
    estimate = build_cost_estimate(
        refreshed, session_store.list_video_segments(source["source_id"]), config.get("billing_tier", "free")
    )
    session_store.update_video_source(source["source_id"], cost_estimate=estimate)


def _render_cue_review(
    source: dict, cues: list[dict], config: dict, worker_path: str, project_dir: str,
) -> None:
    if not cues:
        return
    review_count = sum(
        1 for cue in cues if cue.get("verification_status") in {"needs_review", "recheck_ready"}
    )
    label = f"Kiểm tra và sửa script ({review_count} dòng cần xem lại)"
    with st.expander(label, expanded=review_count > 0):
        only_review = st.checkbox(
            "Chỉ hiện dòng cần kiểm tra", value=review_count > 0,
            key=f"video_only_review_{source['source_id']}",
        )
        visible = [
            cue for cue in cues
            if not only_review or cue.get("verification_status") in {"needs_review", "recheck_ready"}
        ]
        for cue in visible:
            cue_id = str(cue["cue_id"])
            quality = str(cue.get("confidence") or "unknown")
            status = str(cue.get("verification_status") or "unverified")
            st.markdown(
                f"**{format_timestamp(cue.get('start_seconds', 0))} · {cue.get('language', 'unknown')}** "
                f"· độ tin cậy `{quality}` · `{status}`"
            )
            if cue.get("uncertainty_reason"):
                st.warning(str(cue["uncertainty_reason"]))
            edited = st.text_area(
                "Lời thoại", value=str(cue.get("source_text") or ""),
                key=f"video_cue_text_{cue_id}", height=80,
            )
            start_col, end_col = st.columns(2)
            start_value = start_col.number_input(
                "Bắt đầu", min_value=0.0, value=float(cue.get("start_seconds", 0) or 0),
                step=0.1, key=f"video_cue_start_{cue_id}",
            )
            end_value = end_col.number_input(
                "Kết thúc", min_value=0.0, value=float(cue.get("end_seconds", 0) or 0),
                step=0.1, key=f"video_cue_end_{cue_id}",
            )
            save_col, recheck_col = st.columns(2)
            if save_col.button("Lưu sửa", key=f"save_video_cue_{cue_id}", use_container_width=True):
                cleaned = edited.strip()
                if not cleaned:
                    st.error("Lời thoại không được để trống.")
                elif float(end_value) < float(start_value):
                    st.error("Thời gian kết thúc phải sau thời gian bắt đầu.")
                else:
                    changed_text = cleaned != str(cue.get("source_text") or "")
                    session_store.update_video_cue(
                        cue_id, source_text=cleaned, start_seconds=float(start_value), end_seconds=float(end_value),
                        original_source_text=cue.get("original_source_text") or cue.get("source_text") or "",
                        confidence="user", verification_status="verified_user", uncertainty_reason="",
                        revision=int(cue.get("revision", 0) or 0) + 1, recheck={},
                        translation_vi="" if changed_text else cue.get("translation_vi", ""),
                        translation_provider="" if changed_text else cue.get("translation_provider", ""),
                        status="translation_pending" if changed_text else cue.get("status", "translated"),
                    )
                    _refresh_transcript_after_cue_change(source, config)
                    st.rerun()
            if recheck_col.button("Nghe lại đoạn này", key=f"recheck_video_cue_{cue_id}", use_container_width=True):
                _queue_cue_recheck(source, cue_id, int(cue.get("revision", 0) or 0), worker_path, project_dir)
                st.rerun()
            proposal = cue.get("recheck") if isinstance(cue.get("recheck"), dict) else {}
            if proposal.get("source_text"):
                st.markdown("**Đề xuất sau khi AI nghe lại**")
                st.code(str(proposal["source_text"]), language=None)
                st.caption(
                    f"{format_timestamp(proposal.get('start_seconds', 0))}–"
                    f"{format_timestamp(proposal.get('end_seconds', 0))} · "
                    f"độ tin cậy {proposal.get('confidence', 'unknown')}"
                )
                accept_col, keep_col = st.columns(2)
                if accept_col.button("Dùng đề xuất", key=f"accept_video_cue_{cue_id}", type="primary", use_container_width=True):
                    changed_text = str(proposal["source_text"]).strip() != str(cue.get("source_text") or "")
                    session_store.update_video_cue(
                        cue_id, source_text=str(proposal["source_text"]).strip(),
                        start_seconds=float(proposal.get("start_seconds", cue.get("start_seconds", 0)) or 0),
                        end_seconds=float(proposal.get("end_seconds", cue.get("end_seconds", 0)) or 0),
                        speaker=str(proposal.get("speaker") or cue.get("speaker") or ""),
                        language=str(proposal.get("language") or cue.get("language") or "unknown"),
                        original_source_text=cue.get("original_source_text") or cue.get("source_text") or "",
                        confidence=str(proposal.get("confidence") or "unknown"), verification_status="verified_user",
                        uncertainty_reason=str(proposal.get("uncertainty_reason") or ""),
                        revision=int(cue.get("revision", 0) or 0) + 1, recheck={},
                        translation_vi="" if changed_text else cue.get("translation_vi", ""),
                        translation_provider="" if changed_text else cue.get("translation_provider", ""),
                        status="translation_pending" if changed_text else cue.get("status", "translated"),
                    )
                    _refresh_transcript_after_cue_change(source, config)
                    st.rerun()
                if keep_col.button("Giữ bản hiện tại", key=f"keep_video_cue_{cue_id}", use_container_width=True):
                    session_store.update_video_cue(
                        cue_id, recheck={}, verification_status="verified_user",
                        uncertainty_reason="Người dùng đã kiểm tra và giữ bản hiện tại.",
                    )
                    st.rerun()
            st.divider()

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
                "MP4, MOV, WEBM, MPEG hoặc AVI (tối đa 100 MB, 30 phút)",
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
                        media = probe_video_metadata(str(local_path))
                        validate_video_duration(media)
                        duration = float(media["duration_seconds"])
                    except Exception:
                        local_path.unlink(missing_ok=True)
                        raise
                    document = _create_video_document(Path(uploaded.name).stem)
                    session_store.create_video_source(
                        document["document_id"], "upload", file_name=uploaded.name,
                        mime_type=checked["mime_type"], local_path=str(local_path),
                        duration_seconds=duration,
                        metadata={"title": Path(uploaded.name).stem, "media": media},
                        status="awaiting_transcription_confirmation",
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
    if st.button("Phân tích học sâu toàn video", type="primary", use_container_width=True):
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
            job_kind="video_deep_analysis", source_id=source["source_id"], stage="pending",
        )
        session_store.save_analysis_version(version["version_id"], status="running", job_id=job_id)
        session_store.update_video_source(source["source_id"], status="analyzing", error="")
        st.session_state.working_version_id = version["version_id"]
        _start_worker(worker_path, project_dir, job_id, document_id=source["document_id"])
        st.rerun()


def _render_segment(
    segment: dict, source: dict, worker_path: str, project_dir: str, editable: bool = True,
) -> None:
    analysis = normalize_video_segment_result(segment.get("analysis"))
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
        can_visual = bool(
            source.get("source_url")
            or (source.get("ingest_usage") or {}).get("gemini_file_uri")
            or (source.get("local_path") and Path(str(source.get("local_path"))).exists())
        )
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
            _start_worker(worker_path, project_dir, job_id, document_id=source["document_id"])
            st.rerun()


def _video_json_bytes(source: dict, cues: list[dict], segments: list[dict], analysis: dict) -> bytes:
    payload = {
        "schema_version": "video-2.0",
        "source": {**source, "local_path": None},
        "transcript_raw": source.get("raw_transcript") or [],
        "transcript_clean": source.get("clean_transcript") or [],
        "video_cues": cues,
        "segments": segments,
        "analysis": analysis,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _render_usage_cost_breakdown(source: dict, analysis: dict, config: dict) -> None:
    costs = build_video_usage_cost_breakdown(
        source, analysis, config.get("billing_tier", "free")
    )
    rate = float(config.get("usd_to_jpy", 155.0) or 155.0)
    labels = {
        "transcript_primary": "Chép lời chính",
        "transcript_verification": "Xác minh transcript",
        "translation_vi": "Dịch tiếng Việt",
        "deep_analysis": "Phân tích học sâu",
    }
    rows = []
    for key, label in labels.items():
        cost = costs.get(key) or {}
        rows.append({
            "Giai đoạn": label,
            "Token vào": int(cost.get("input_tokens", 0) or 0),
            "Token ra": int(cost.get("output_tokens", 0) or 0),
            "Chi phí JPY": round(float(cost.get("paid_equivalent_usd", 0) or 0) * rate, 3),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_results(source: dict, cues: list[dict], segments: list[dict], analysis: dict, config: dict) -> None:
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
    _render_usage_cost_breakdown(source, analysis, config)
    cols = st.columns(3)
    cols[0].download_button(
        "Tải Markdown", markdown.encode("utf-8"), "video_analysis.md", "text/markdown", use_container_width=True
    )
    cols[1].download_button(
        "Tải Word", export_to_docx(markdown, "video_analysis.docx"), "video_analysis.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True,
    )
    cols[2].download_button(
        "Tải JSON đầy đủ", _video_json_bytes(source, cues, segments, analysis), "video_analysis.json",
        "application/json", use_container_width=True,
    )


def render_video_tab(
    *, config: dict, active_document: dict, worker_path: str, project_dir: str,
) -> None:
    """Render source creation, two confirmations, progress, and completed segments."""
    workspace = session_store.get_document_workspace(active_document.get("document_id", "")) or {}
    source = workspace.get("video_source") if active_document.get("document_type") == "video" else None
    segments = workspace.get("video_segments") or []
    cues = workspace.get("video_cues") or []
    if source is None:
        _render_new_source(config, worker_path, project_dir)
        return

    title = (source.get("metadata") or {}).get("title") or source.get("file_name") or "Video"
    st.subheader(title)
    if source.get("source_kind") == "youtube" and source.get("video_id") and cues:
        render_youtube_player(str(source["video_id"]), cues, key=f"youtube_sync_{source['source_id']}")
    elif source.get("source_url"):
        st.video(source["source_url"])
    elif source.get("local_path") and Path(source["local_path"]).exists():
        if cues:
            native_container_key = f"upload_media_{str(source['source_id']).replace('-', '_')}"
            with st.container(key=native_container_key):
                st.video(source["local_path"])
                render_upload_player(
                    cues, key=f"upload_sync_{source['source_id']}",
                    native_container_key=native_container_key, source_id=str(source["source_id"]),
                )
        else:
            st.video(source["local_path"])
    elif source.get("source_kind") == "upload":
        st.warning("File video tạm đã hết hạn. Kết quả script vẫn còn nhưng cần tải lại file để phát video.")
    st.caption(
        f"Nguồn: {'YouTube' if source.get('source_kind') == 'youtube' else 'File tải lên'} | "
        f"Thời lượng: {format_timestamp(source.get('duration_seconds', 0))} | "
        f"Transcript: {source.get('transcript_provider') or 'chưa có'}"
    )
    status = str(source.get("status") or "pending")
    pipeline_version = int((source.get("metadata") or {}).get("transcript_pipeline_version", 0) or 0)
    if cues and pipeline_version < 2:
        st.warning("Transcript Legacy / chưa xác minh. Bạn vẫn có thể xem và sửa từng dòng.")
        can_upgrade = bool(
            source.get("source_kind") == "upload"
            and source.get("local_path")
            and Path(str(source.get("local_path"))).exists()
        )
        if can_upgrade and st.button(
            "Nâng cấp transcript sang V2", key=f"upgrade_video_v2_{source['source_id']}",
            use_container_width=True,
        ):
            metadata = {
                **(source.get("metadata") or {}),
                "upgrade_v2_requested": True,
                "upgrade_v2_started": False,
                "transcription_completed_windows": [],
            }
            session_store.update_video_source(source["source_id"], metadata=metadata)
            _queue_video_job(source, config, worker_path, project_dir, "video_transcribe")
            st.rerun()
    if cues and status not in {"transcribing", "translating", "ingesting", "segmenting", "analyzing"}:
        _render_cue_review(source, cues, config, worker_path, project_dir)
    elif cues:
        st.caption("Tạm khóa chỉnh script trong khi job nền đang cập nhật dữ liệu.")

    if status in {"pending", "ingesting", "segmenting"}:
        st.info("Đang lấy caption và tạo mục lục. Bạn có thể chuyển sang bài khác; job vẫn tiếp tục.")
        return
    if status == "transcribing":
        metadata = source.get("metadata") or {}
        done = len(metadata.get("transcription_completed_windows") or [])
        total = int(metadata.get("transcription_total_windows") or 1)
        st.info(f"Đang tạo script: {done}/{total} cửa sổ audio đã hoàn thành.")
        st.progress(min(1.0, done / max(1, total)))
        return
    if status == "translating":
        translated = sum(1 for cue in cues if cue.get("translation_vi"))
        st.info(f"Đang dịch caption: {translated}/{len(cues)} dòng đã có bản dịch Việt.")
        st.progress(translated / max(1, len(cues)))
        return
    if status == "caption_unavailable":
        error_code = str((source.get("metadata") or {}).get("caption_error_code") or "")
        if error_code == "youtube_ip_blocked":
            st.error("YouTube đang chặn IP máy chủ Streamlit Cloud, nên app chưa thể lấy caption tự động.")
            st.caption(
                "Link vẫn hợp lệ. Đây là giới hạn của YouTube với IP cloud, không phải lỗi Gemini hay lỗi video."
            )
        elif error_code == "youtube_caption_unavailable":
            st.error("Video không có caption tiếng Nhật hoặc tiếng Anh công khai.")
        else:
            st.error("Chưa thể lấy caption từ YouTube ở thời điểm này.")
        with st.expander("Dán subtitle hoặc transcript để tiếp tục", expanded=True):
            st.caption(
                "Dán SRT/VTT để giữ timestamp và đồng bộ player. Nếu chỉ có văn bản, hãy để mỗi câu một dòng; "
                "app sẽ tạo timestamp ước lượng để vẫn phân tích được."
            )
            manual_text = st.text_area(
                "Subtitle / transcript", key=f"manual_youtube_transcript_{source['source_id']}",
                height=240, placeholder="00:00:01,000 --> 00:00:03,000\nHello everyone.",
            )
            if st.button(
                "Dùng transcript này", type="primary", use_container_width=True,
                key=f"apply_manual_youtube_transcript_{source['source_id']}",
            ):
                try:
                    _apply_manual_youtube_transcript(source, manual_text, config)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        st.info(
            "Không dùng proxy hoặc cookie YouTube trong app: chúng không ổn định trên Cloud và có thể làm tài khoản "
            "YouTube bị chặn. Bạn cũng có thể tải file video từ thiết bị vào tab File video để app tự tạo script."
        )
        st.divider()
        _render_new_source(config, worker_path, project_dir)
        return
    if status == "awaiting_range_selection":
        st.warning("Video dài hơn 30 phút. Hãy chọn một khoảng tối đa 30 phút để tạo bài học.")
        range_start, range_end = _range_inputs(source, f"caption_range_{source['source_id']}")
        if st.button("Dùng khoảng đã chọn", type="primary", use_container_width=True):
            try:
                _apply_caption_range(source, range_start, range_end, config)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        return
    if status in {"awaiting_ingest_confirmation", "awaiting_transcription_confirmation"}:
        estimate = estimate_audio_transcription_cost(
            float(source.get("duration_seconds") or 0), config.get("billing_tier", "free")
        )
        rate = float(config.get("usd_to_jpy", 155) or 155)
        cols = st.columns(4)
        cols[0].metric("Token audio chính", f"{int(estimate.get('input_tokens', 0)):,}")
        cols[1].metric("Cửa sổ 90 giây", str(estimate.get("window_count", 0)))
        cols[2].metric(
            "Chép lời chính",
            f"¥{float((estimate.get('primary_expected') or {}).get('paid_equivalent_usd', 0)) * rate:,.2f}",
        )
        cols[3].metric(
            "Xác minh dự kiến",
            f"¥{float((estimate.get('verification_expected') or {}).get('paid_equivalent_usd', 0)) * rate:,.2f}",
        )
        st.caption(
            "Tổng dự kiến "
            f"¥{float((estimate.get('expected') or {}).get('paid_equivalent_usd', 0)) * rate:,.2f}; "
            "mức tối đa gồm xác minh có giới hạn "
            f"¥{float((estimate.get('maximum') or {}).get('paid_equivalent_usd', 0)) * rate:,.2f}. "
            "App gửi audio FLAC mono 16 kHz cho Gemini Flash; dịch tiếng Việt được tính riêng sau khi script hoàn tất."
        )
        if st.button("Tạo script", type="primary", use_container_width=True):
            _queue_video_job(source, config, worker_path, project_dir, "video_transcribe")
            st.rerun()
        return
    if status in {"transcription_partial"}:
        st.warning(source.get("error") or "Một số cửa sổ audio chưa tạo được script.")
        st.caption("Các cửa sổ đã hoàn thành vẫn được giữ. Chạy tiếp chỉ xử lý phần còn thiếu.")
        if st.button("Tiếp tục tạo script", type="primary", use_container_width=True):
            _queue_video_job(source, config, worker_path, project_dir, "video_transcribe")
            st.rerun()
        return
    if status in {"awaiting_translation_confirmation", "translation_partial"}:
        pending = [cue for cue in cues if not str(cue.get("translation_vi") or "").strip()]
        estimate = estimate_cue_translation_cost(pending, config.get("billing_tier", "free"))
        rate = float(config.get("usd_to_jpy", 155) or 155)
        if status == "translation_partial" and source.get("error"):
            st.warning(source["error"])
        st.info(f"Đã có script. {len(pending)}/{len(cues)} dòng chưa có bản dịch tiếng Việt.")
        cols = st.columns(3)
        cols[0].metric("Dòng cần dịch", len(pending))
        cols[1].metric("Số batch", estimate.get("batch_count", 0))
        cols[2].metric(
            "Chi phí tương đương",
            f"¥{float((estimate.get('expected') or {}).get('paid_equivalent_usd', 0)) * rate:,.2f}",
        )
        if st.button("Dịch các dòng còn thiếu", type="primary", use_container_width=True):
            _queue_video_job(source, config, worker_path, project_dir, "video_translate")
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
            _render_results(source, cues, selected_segments, analysis, config)
            if selected_segments is not segments:
                segments = selected_segments
    elif status == "failed":
        st.error(source.get("error") or "Phân tích segment chưa hoàn tất.")
        if source.get("source_kind") == "upload" and not cues:
            st.caption("File video vẫn được giữ. Bạn có thể chạy lại bước tạo script từ cửa sổ đầu tiên.")
            if st.button("Chạy lại tạo script", type="primary", use_container_width=True):
                _queue_video_job(source, config, worker_path, project_dir, "video_transcribe")
                st.rerun()
        elif segments and st.button("Chạy lại phân tích từ transcript hiện có", type="primary", use_container_width=True):
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
