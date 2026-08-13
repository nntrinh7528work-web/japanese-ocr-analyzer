"""Standalone worker — run analysis in isolated process, save to DB."""

import sys
import json
import subprocess
from pathlib import Path
import config as app_config
from modules.job_store import update_job, get_job
from modules import session_store
from modules.job_workflow import items_source_hash
from modules.notion_sync import enqueue_analysis_sync, notion_connection_state
from modules.text_analyzer import run_analysis, run_page_analyses
from modules.sentence_analyzer import analyze_manual_sentence
from modules.translation_guidance import analyze_guidance_job
from modules.sentence_analyzer import merge_manual_breakdown
from modules.translation_guidance import merge_guidance_job
from modules.video_analyzer import (
    analyze_video_segment_batch,
    analyze_video_visual_segment,
    build_cost_estimate,
    build_audio_windows,
    build_cue_translation_batches,
    build_segment_batches,
    build_segments,
    build_video_analysis,
    clean_transcript,
    classify_youtube_caption_error,
    cues_to_transcript_rows,
    fetch_youtube_caption,
    merge_transcript_cues,
    normalize_transcript,
    transcribe_audio_window,
    transcript_rows_to_cues,
    transcript_hash,
    translate_video_cue_batch,
)

MAX_VIDEO_DURATION_SECONDS = int(getattr(app_config, "MAX_VIDEO_DURATION_SECONDS", 30 * 60))

def _mapping(value: object) -> dict:
    """Treat malformed persisted/model values as empty structured data."""
    return dict(value) if isinstance(value, dict) else {}


def _has_sentence_breakdown(segment: dict) -> bool:
    return bool(_mapping(segment.get("analysis")).get("sentence_breakdown"))


def _video_notion_items(source: dict, segments: list[dict]) -> list[dict]:
    """Expose only the cleaned transcript to Notion and shared exporters."""
    return [
        {
            "id": segment.get("segment_id"),
            "name": segment.get("title") or f"Đoạn {segment.get('ordinal', 0)}",
            "edited_text": segment.get("clean_text") or "",
            "detected_language": segment.get("language") or "unknown",
            "ocr_result": {"ocr_notes": [], "confidence": None, "text_direction": "horizontal"},
        }
        for segment in segments
    ]


def _dispatch_notion_for_video(job_data: dict, source: dict, segments: list[dict], analysis: dict) -> None:
    settings = session_store.load_settings(job_data["session_id"])
    if not settings.get("auto_notion_sync", True) or not notion_connection_state()["configured"]:
        return
    notion_run = enqueue_analysis_sync(
        job_data["session_id"], _video_notion_items(source, segments), analysis,
        billing_tier=settings.get("billing_tier", "free"),
        usd_to_jpy=float(settings.get("usd_to_jpy", 155)),
        document_id=job_data["document_id"],
    )
    if session_store.dispatch_notion_sync_run(notion_run["run_id"]):
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parent / "notion_worker.py"), notion_run["run_id"]],
            stdout=subprocess.DEVNULL,
            stderr=open(str(Path(__file__).resolve().parent / "notion_worker_error.log"), "a"),
            cwd=str(Path(__file__).resolve().parent),
        )


def _finalize_video_transcript(
    job_id: str, job_data: dict, source: dict, cues: list[dict], provider: str,
    usage: dict, billing_tier: str, warnings: list[str] | None = None,
) -> None:
    raw_rows = cues_to_transcript_rows(cues)
    clean_rows, cleanup_warnings = clean_transcript(raw_rows)
    duration = float(source.get("duration_seconds") or (raw_rows[-1]["end"] if raw_rows else 0))
    current_hash = transcript_hash(raw_rows)
    all_warnings = [*(warnings or []), *(usage.get("warnings") or []), *cleanup_warnings]
    metadata = {**(source.get("metadata") or {})}
    selected = metadata.get("range_start") is not None and metadata.get("range_end") is not None
    if duration > MAX_VIDEO_DURATION_SECONDS and not selected:
        status = "awaiting_range_selection"
        session_store.update_video_source(
            source["source_id"], duration_seconds=duration, raw_transcript=raw_rows,
            clean_transcript=[], transcript_warnings=all_warnings, metadata=metadata,
            transcript_provider=provider, transcript_hash=current_hash, ingest_usage=usage,
            status=status, error="",
        )
        session_store.update_document_source_hash(job_data["document_id"], current_hash, status=status)
        update_job(
            job_id, "done", stage=status,
            result={"job_kind": job_data.get("job_kind"), "status": status, "duration_seconds": duration},
        )
        return

    segments = build_segments(clean_rows, namespace=source["source_id"])
    session_store.replace_video_segments(source["source_id"], segments)
    translated = all(str(cue.get("translation_vi") or "").strip() for cue in cues)
    status = "awaiting_cost_confirmation" if translated else "awaiting_translation_confirmation"
    session_store.update_video_source(
        source["source_id"], duration_seconds=duration, raw_transcript=raw_rows,
        clean_transcript=clean_rows, transcript_warnings=all_warnings, metadata=metadata,
        transcript_provider=provider, transcript_hash=current_hash, ingest_usage=usage,
        status=status, error="",
    )
    session_store.update_document_source_hash(job_data["document_id"], current_hash, status=status)
    refreshed = session_store.get_video_source(source["source_id"]) or source
    estimate = build_cost_estimate(refreshed, segments, billing_tier)
    session_store.update_video_source(source["source_id"], cost_estimate=estimate, status=status)
    update_job(
        job_id, "done", stage=status,
        result={
            "job_kind": job_data.get("job_kind"), "status": status,
            "source_id": source["source_id"], "cue_count": len(cues),
            "segment_count": len(segments), "cost_estimate": estimate,
        },
    )


def _run_video_ingest(job_id: str, job_data: dict, payload: dict) -> None:
    source = session_store.get_video_source(job_data.get("source_id"))
    if not source:
        raise ValueError("Không tìm thấy nguồn video của job.")
    update_job(job_id, "running", stage="fetching_caption", checkpoint={"window": 0})
    if source.get("source_kind") != "youtube":
        raise ValueError("Job lấy caption chỉ dành cho link YouTube.")
    try:
        rows, provider = fetch_youtube_caption(
            str(source.get("video_id") or ""), str(payload.get("preferred_language") or "unknown")
        )
    except Exception as exc:
        error_code = classify_youtube_caption_error(exc)
        metadata = {**(source.get("metadata") or {}), "caption_error_code": error_code}
        session_store.update_video_source(
            source["source_id"], status="caption_unavailable", metadata=metadata, error=str(exc)
        )
        update_job(
            job_id, "done", stage="caption_unavailable",
            result={
                "job_kind": "video_ingest", "status": "caption_unavailable",
                "reason": str(exc), "error_code": error_code,
            },
        )
        return
    cues = transcript_rows_to_cues(rows, source["source_id"], provider)
    session_store.replace_video_cues(source["source_id"], cues)
    _finalize_video_transcript(
        job_id, job_data, source, cues, provider, {}, payload.get("billing_tier", "free")
    )


def _cue_recheck_window(source: dict, cue: dict) -> dict:
    duration = float(source.get("duration_seconds") or 0)
    cue_start = float(cue.get("start_seconds", 0) or 0)
    cue_end = max(cue_start, float(cue.get("end_seconds", cue_start) or cue_start))
    center = (cue_start + cue_end) / 2
    start = max(0.0, center - 12.5)
    end = min(duration or center + 12.5, start + 25.0)
    if end - start < 12 and duration:
        start = max(0.0, end - 12)
    return {"index": -1, "start": start, "end": end}


def _transcribe_cue_proposal(source: dict, cue: dict) -> tuple[dict, dict]:
    window = _cue_recheck_window(source, cue)
    rows, usage = transcribe_audio_window(source, window, str(Path(source["local_path"]).parent))
    cue_start = float(cue.get("start_seconds", 0) or 0)
    cue_end = float(cue.get("end_seconds", cue_start) or cue_start)
    matching = [
        row for row in rows
        if float(row.get("end", 0)) >= cue_start - 1.5
        and float(row.get("start", 0)) <= cue_end + 1.5
    ]
    if not matching:
        midpoint = (cue_start + cue_end) / 2
        matching = [min(rows, key=lambda row: abs(((float(row["start"]) + float(row["end"])) / 2) - midpoint))]
    confidence_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    confidence = min(
        (str(row.get("confidence") or "unknown") for row in matching),
        key=lambda value: confidence_rank.get(value, 0),
        default="unknown",
    )
    proposal = {
        "source_text": " ".join(str(row.get("text") or "").strip() for row in matching).strip(),
        "start_seconds": min(float(row.get("start", cue_start)) for row in matching),
        "end_seconds": max(float(row.get("end", cue_end)) for row in matching),
        "speaker": next((str(row.get("speaker") or "") for row in matching if row.get("speaker")), ""),
        "language": str(matching[0].get("language") or cue.get("language") or "unknown"),
        "confidence": confidence,
        "uncertainty_reason": "; ".join(
            str(row.get("uncertainty_reason") or "").strip()
            for row in matching if str(row.get("uncertainty_reason") or "").strip()
        ),
        "window": window,
    }
    return proposal, usage


def _run_video_cue_recheck(job_id: str, job_data: dict, payload: dict) -> None:
    source = session_store.get_video_source(job_data.get("source_id"))
    if not source:
        raise ValueError("Không tìm thấy nguồn video cần nghe lại.")
    expected_hash = str(job_data.get("source_hash") or "")
    if expected_hash and expected_hash != str(source.get("transcript_hash") or ""):
        raise ValueError("Transcript đã thay đổi; kết quả nghe lại cũ bị từ chối.")
    cue_id = str(payload.get("cue_id") or "")
    cue = next((row for row in session_store.list_video_cues(source["source_id"]) if row.get("cue_id") == cue_id), None)
    if not cue:
        raise ValueError("Không tìm thấy dòng script cần nghe lại.")
    expected_revision = int(payload.get("revision", cue.get("revision", 0)) or 0)
    if expected_revision != int(cue.get("revision", 0) or 0):
        raise ValueError("Dòng script đã được sửa; hãy bấm nghe lại lần nữa.")
    update_job(job_id, "running", stage="rechecking_cue", checkpoint={"cue_id": cue_id})
    proposal, usage = _transcribe_cue_proposal(source, cue)
    refreshed = session_store.get_video_source(source["source_id"]) or {}
    latest_cue = next(
        (row for row in session_store.list_video_cues(source["source_id"]) if row.get("cue_id") == cue_id),
        None,
    )
    if (
        (expected_hash and expected_hash != str(refreshed.get("transcript_hash") or ""))
        or not latest_cue
        or expected_revision != int(latest_cue.get("revision", 0) or 0)
    ):
        raise ValueError("Transcript đã thay đổi trong lúc nghe lại; đề xuất cũ bị từ chối.")
    session_store.update_video_cue(
        cue_id, recheck=proposal, verification_status="recheck_ready",
        uncertainty_reason=proposal.get("uncertainty_reason") or cue.get("uncertainty_reason") or "",
    )
    usage_state = dict(refreshed.get("ingest_usage") or {})
    runs = list(usage_state.get("runs") or [])
    run_id = f"{source['source_id']}:manual-verify:{cue_id}:{expected_revision}"
    if run_id not in {str(run.get("run_id")) for run in runs if isinstance(run, dict)}:
        runs.append({
            "run_id": run_id, "model_used": usage.get("model_used"),
            "usage": usage, "modality": "audio", "stage": "transcript_verification_manual",
        })
    cues.sort(
        key=lambda row: (
            float(row.get("start_seconds", 0) or 0),
            float(row.get("end_seconds", 0) or 0),
        )
    )
    usage_state["runs"] = runs
    usage_state["input_tokens"] = sum(
        int((run.get("usage") or {}).get("input_tokens", 0))
        for run in runs if isinstance(run, dict)
    )
    usage_state["output_tokens"] = sum(
        int((run.get("usage") or {}).get("output_tokens", 0))
        for run in runs if isinstance(run, dict)
    )
    session_store.update_video_source(source["source_id"], ingest_usage=usage_state)
    update_job(
        job_id, "done", stage="done",
        result={"job_kind": "video_cue_recheck", "cue_id": cue_id, "proposal": proposal, "usage": usage},
    )

def _auto_verify_transcript(source: dict, cues: list[dict], usage_state: dict) -> tuple[list[dict], dict]:
    """Recheck only the riskiest fifth of cues and uncovered audible regions."""
    candidates = [row for row in cues if row.get("verification_status") == "needs_review"]
    covered_gaps: list[dict] = []
    for run in usage_state.get("runs") or []:
        usage = run.get("usage") if isinstance(run, dict) else {}
        for gap in (usage or {}).get("coverage_gaps") or []:
            if not isinstance(gap, dict):
                continue
            start = float(gap.get("start", 0) or 0)
            end = float(gap.get("end", start) or start)
            if end <= start or any(
                float(cue.get("end_seconds", 0) or 0) >= start
                and float(cue.get("start_seconds", 0) or 0) <= end
                for cue in cues
            ):
                continue
            covered_gaps.append({
                "cue_id": f"gap-{source['source_id']}-{start:.3f}-{end:.3f}",
                "start_seconds": start, "end_seconds": end,
                "source_text": "", "confidence": "unknown", "revision": 0,
                "verification_status": "needs_review", "_coverage_gap": True,
            })
    candidates.extend(covered_gaps)
    limit = min(8, max(0, (max(len(cues), len(candidates)) + 4) // 5))
    runs = list(usage_state.get("runs") or [])
    existing_ids = {str(run.get("run_id")) for run in runs if isinstance(run, dict)}
    confidence_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    for cue in candidates[:limit]:
        run_id = f"{source['source_id']}:verify:{cue['cue_id']}:{int(cue.get('revision', 0) or 0)}"
        if run_id in existing_ids:
            continue
        try:
            proposal, usage = _transcribe_cue_proposal(source, cue)
            current_rank = confidence_rank.get(str(cue.get("confidence") or "unknown"), 0)
            proposed_rank = confidence_rank.get(str(proposal.get("confidence") or "unknown"), 0)
            if cue.get("_coverage_gap") and proposal.get("source_text"):
                cue.update({
                    "source_text": proposal["source_text"],
                    "original_source_text": proposal["source_text"],
                    "start_seconds": proposal["start_seconds"],
                    "end_seconds": proposal["end_seconds"],
                    "speaker": proposal.get("speaker") or "",
                    "language": proposal.get("language") or "unknown",
                    "confidence": proposal.get("confidence") or "unknown",
                    "verification_status": "verified_auto" if proposed_rank >= 3 else "recheck_ready",
                    "uncertainty_reason": "" if proposed_rank >= 3 else (
                        proposal.get("uncertainty_reason") or "Đoạn lời nói bổ sung cần kiểm tra."
                    ),
                    "revision": 1,
                    "recheck": {} if proposed_rank >= 3 else proposal,
                    "translation_vi": "",
                    "translation_provider": "",
                    "status": "translation_pending",
                    "transcript_provider": "gemini_audio_v2_recheck",
                })
                cue.pop("_coverage_gap", None)
                cues.append(cue)
            elif proposed_rank >= 3 and proposed_rank > current_rank and proposal.get("source_text"):
                changed_text = proposal["source_text"] != str(cue.get("source_text") or "")
                cue["source_text"] = proposal["source_text"]
                cue["start_seconds"] = proposal["start_seconds"]
                cue["end_seconds"] = proposal["end_seconds"]
                cue["speaker"] = proposal.get("speaker") or cue.get("speaker") or ""
                cue["language"] = proposal.get("language") or cue.get("language") or "unknown"
                cue["confidence"] = proposal["confidence"]
                cue["verification_status"] = "verified_auto"
                cue["uncertainty_reason"] = ""
                cue["revision"] = int(cue.get("revision", 0) or 0) + 1
                cue["recheck"] = {}
                if changed_text:
                    cue["translation_vi"] = ""
                    cue["translation_provider"] = ""
                    cue["status"] = "translation_pending"
            else:
                cue["recheck"] = proposal
                cue["verification_status"] = "recheck_ready"
            runs.append({"run_id": run_id, "model_used": usage.get("model_used"), "usage": usage, "modality": "audio", "stage": "transcript_verification"})
            existing_ids.add(run_id)
        except Exception as exc:
            cue["uncertainty_reason"] = str(cue.get("uncertainty_reason") or exc)
    cues.sort(
        key=lambda row: (
            float(row.get("start_seconds", 0) or 0),
            float(row.get("end_seconds", 0) or 0),
        )
    )
    usage_state["runs"] = runs
    usage_state["input_tokens"] = sum(
        int((run.get("usage") or {}).get("input_tokens", 0)) for run in runs if isinstance(run, dict)
    )
    usage_state["output_tokens"] = sum(
        int((run.get("usage") or {}).get("output_tokens", 0)) for run in runs if isinstance(run, dict)
    )
    return cues, usage_state

def _run_video_transcribe(job_id: str, job_data: dict, payload: dict) -> None:
    source = session_store.get_video_source(job_data.get("source_id"))
    if not source:
        raise ValueError("Không tìm thấy file video của job.")
    windows = build_audio_windows(float(source.get("duration_seconds") or 0))
    if not windows:
        raise ValueError("Không đọc được thời lượng video để tạo script.")
    metadata = dict(source.get("metadata") or {})
    completed = {int(value) for value in metadata.get("transcription_completed_windows") or []}
    cues = session_store.list_video_cues(source["source_id"])
    if metadata.get("upgrade_v2_requested") and not metadata.get("upgrade_v2_started"):
        metadata["legacy_transcript_backup"] = cues
        metadata["upgrade_v2_started"] = True
        metadata["transcription_completed_windows"] = []
        completed = set()
        cues = []
        session_store.replace_video_cues(source["source_id"], [])
        session_store.update_video_source(source["source_id"], metadata=metadata)
    usage_state = dict(source.get("ingest_usage") or {})
    runs = list(usage_state.get("runs") or [])
    run_ids = {str(run.get("run_id")) for run in runs if isinstance(run, dict)}
    failures = []
    session_store.update_video_source(source["source_id"], status="transcribing", error="")

    for window in windows:
        index = int(window["index"])
        if index in completed:
            continue
        update_job(
            job_id, "running", stage="transcribing_audio",
            checkpoint={"window": index, "completed": len(completed), "total": len(windows)},
        )
        try:
            rows, usage = transcribe_audio_window(source, window, str(Path(source["local_path"]).parent))
            incoming = transcript_rows_to_cues(rows, source["source_id"], "gemini_audio")
            cues = merge_transcript_cues(cues, incoming)
            session_store.replace_video_cues(source["source_id"], cues)
            completed.add(index)
            run_id = f"{source['source_id']}:audio-v2:{index}"
            if run_id not in run_ids:
                runs.append({
                    "run_id": run_id, "model_used": usage.get("model_used"),
                    "usage": usage, "modality": "audio", "stage": "transcript_primary",
                })
                run_ids.add(run_id)
            metadata["transcription_completed_windows"] = sorted(completed)
            metadata["transcription_total_windows"] = len(windows)
            usage_state["runs"] = runs
            usage_state["input_tokens"] = sum(int((run.get("usage") or {}).get("input_tokens", 0)) for run in runs)
            usage_state["output_tokens"] = sum(int((run.get("usage") or {}).get("output_tokens", 0)) for run in runs)
            usage_state["model_used"] = usage.get("model_used")
            session_store.update_video_source(
                source["source_id"], metadata=metadata, ingest_usage=usage_state, status="transcribing"
            )
        except Exception as exc:
            failures.append({"window": index, "error": str(exc)})

    if failures:
        metadata["transcription_errors"] = failures
        if not cues and metadata.get("legacy_transcript_backup"):
            cues = list(metadata["legacy_transcript_backup"])
            session_store.replace_video_cues(source["source_id"], cues)
        status = "transcription_partial" if any(
            cue.get("transcript_provider") == "gemini_audio_v2" for cue in cues
        ) else "failed"
        message = "; ".join(f"Cửa sổ {row['window']}: {row['error']}" for row in failures)
        session_store.update_video_source(
            source["source_id"], metadata=metadata, ingest_usage=usage_state, status=status, error=message
        )
        update_job(
            job_id, "done", stage=status,
            result={"job_kind": "video_transcribe", "status": status, "failures": failures},
        )
        return
    metadata.pop("transcription_errors", None)
    metadata.pop("legacy_transcript_backup", None)
    metadata.pop("upgrade_v2_requested", None)
    metadata.pop("upgrade_v2_started", None)
    metadata["transcript_pipeline_version"] = 2
    metadata["transcript_window_seconds"] = 90
    metadata["transcript_overlap_seconds"] = 5
    source = session_store.get_video_source(source["source_id"]) or source
    source["metadata"] = metadata
    cues = session_store.list_video_cues(source["source_id"])
    cues, usage_state = _auto_verify_transcript(source, cues, usage_state)
    session_store.replace_video_cues(source["source_id"], cues)
    _finalize_video_transcript(
        job_id, job_data, source, cues, "gemini_audio", usage_state,
        payload.get("billing_tier", "free"),
    )


def _run_video_translate(job_id: str, job_data: dict, payload: dict) -> None:
    source = session_store.get_video_source(job_data.get("source_id"))
    if not source:
        raise ValueError("Không tìm thấy nguồn video cần dịch.")
    cues = session_store.list_video_cues(source["source_id"])
    pending = [cue for cue in cues if not str(cue.get("translation_vi") or "").strip()]
    batches = build_cue_translation_batches(pending)
    runs = list(source.get("translation_runs") or [])
    run_ids = {str(run.get("run_id")) for run in runs if isinstance(run, dict)}
    failures = []
    session_store.update_video_source(source["source_id"], status="translating", error="")
    for index, batch in enumerate(batches, 1):
        update_job(
            job_id, "running", stage="translating_cues",
            checkpoint={"batch": index, "completed": index - 1, "total": len(batches)},
        )
        try:
            translated, usage = translate_video_cue_batch(batch)
            for cue in batch:
                session_store.update_video_cue(
                    cue["cue_id"], translation_vi=translated[cue["cue_id"]],
                    translation_provider="gemini_translation", status="translated", warning="",
                )
            run_id = f"{job_id}:translation:{index}"
            if run_id not in run_ids:
                runs.append({"run_id": run_id, "model_used": usage.get("model_used"), "usage": usage})
                run_ids.add(run_id)
            session_store.update_video_source(source["source_id"], translation_runs=runs)
        except Exception as exc:
            failures.append({"batch": index, "error": str(exc)})
    cues = session_store.list_video_cues(source["source_id"])
    rows = cues_to_transcript_rows(cues)
    clean_rows, warnings = clean_transcript(rows)
    if failures:
        message = "; ".join(f"Batch {row['batch']}: {row['error']}" for row in failures)
        session_store.update_video_source(
            source["source_id"], raw_transcript=rows, clean_transcript=clean_rows,
            status="translation_partial", error=message,
        )
        update_job(job_id, "done", stage="translation_partial", result={"failures": failures})
        return
    session_store.update_video_source(
        source["source_id"], raw_transcript=rows, clean_transcript=clean_rows,
        transcript_warnings=[*(source.get("transcript_warnings") or []), *warnings],
        status="awaiting_cost_confirmation", error="", translation_runs=runs,
    )
    update_job(
        job_id, "done", stage="awaiting_cost_confirmation",
        result={"job_kind": "video_translate", "status": "awaiting_cost_confirmation"},
    )


def _hard_sentence_row(segment: dict, candidate: object) -> dict | None:
    if isinstance(candidate, dict):
        text = str(candidate.get("original") or candidate.get("text") or "").strip()
    else:
        text = str(candidate or "").strip()
    if not text:
        return None
    return {
        "sentence_id": f"p{int(segment.get('ordinal') or 1)}-s1",
        "ordinal": 1,
        "original": text,
        "text": text,
        "detected_language": segment.get("language") or "english",
        "language_confidence": 1.0,
        "language_source": "video_segment",
        "complexity_score": 12,
    }


def _run_video_analysis(job_id: str, job_data: dict, payload: dict) -> None:
    source = session_store.get_video_source(job_data.get("source_id"))
    if not source:
        raise ValueError("Không tìm thấy nguồn video của job.")
    segments = session_store.list_video_segments(source["source_id"])
    if not segments:
        raise ValueError("Video chưa có transcript hoặc mục lục để phân tích.")
    deep_count = sum(1 for row in segments if _has_sentence_breakdown(row))
    failures = []
    pending = [row for row in segments if not (row.get("status") == "done" and row.get("analysis"))]
    batches = build_segment_batches(pending)
    completed_count = len(segments) - len(pending)

    def persist_result(segment: dict, result: dict, usage: dict) -> None:
        nonlocal deep_count, completed_count
        result = _mapping(result)
        usage = _mapping(usage)
        candidate = _hard_sentence_row(segment, result.get("hard_sentence_candidate"))
        if candidate and deep_count < 15 and bool(payload.get("analyze_hard_sentences", True)):
            try:
                envelope = analyze_manual_sentence(
                    candidate,
                    segment.get("clean_text") or "",
                    candidate["detected_language"],
                    model_name=payload.get("deep_model"),
                    reasoning_effort=payload.get("reasoning_effort", "standard"),
                )
                result["sentence_breakdown"] = envelope.get("breakdown") or {}
                result["sentence_analysis_usage"] = envelope.get("usage") or {}
                result["sentence_analysis_model"] = envelope.get("model_used")
                usage["deep_sentence_usage"] = envelope.get("usage") or {}
                deep_count += 1
            except Exception as deep_error:
                result["sentence_analysis_error"] = str(deep_error)
        session_store.update_video_segment(
            segment["segment_id"], analysis=result, usage=usage, status="done", error=""
        )
        completed_count += 1
        update_job(
            job_id, "running", stage="analyzing_segments",
            checkpoint={"completed": completed_count, "total": len(segments), "segment_id": segment["segment_id"]},
        )

    for batch_index, batch in enumerate(batches):
        for segment in batch:
            session_store.update_video_segment(segment["segment_id"], status="running", error="")
        update_job(
            job_id, "running", stage="analyzing_segments",
            checkpoint={"completed": completed_count, "total": len(segments), "batch": batch_index + 1},
        )
        try:
            results, batch_usage = analyze_video_segment_batch(batch)
            batch_usage = _mapping(batch_usage)
            batch_usage["run_id"] = f"{job_id}:batch:{batch_index + 1}"
            for segment, result in zip(batch, results):
                persist_result(segment, result, dict(batch_usage))
        except Exception as batch_error:
            # A malformed result for one row must not discard neighboring segments.
            if len(batch) > 1:
                for segment in batch:
                    try:
                        rows, usage = analyze_video_segment_batch([segment])
                        usage = _mapping(usage)
                        usage["run_id"] = f"{job_id}:segment:{segment['segment_id']}"
                        persist_result(segment, rows[0], usage)
                    except Exception as segment_error:
                        failures.append({"segment_id": segment["segment_id"], "error": str(segment_error)})
                        session_store.update_video_segment(segment["segment_id"], status="failed", error=str(segment_error))
                        completed_count += 1
            else:
                segment = batch[0]
                failures.append({"segment_id": segment["segment_id"], "error": str(batch_error)})
                session_store.update_video_segment(segment["segment_id"], status="failed", error=str(batch_error))
                completed_count += 1

    segments = session_store.list_video_segments(source["source_id"])
    analysis = build_video_analysis(
        {**source, "video_cues": session_store.list_video_cues(source["source_id"])}, segments
    )
    analysis["video_segment_errors"] = failures
    if not analysis.get("video_segments"):
        raise RuntimeError("Không có đoạn video nào phân tích thành công.")
    version_id = job_data.get("version_id")
    if not version_id:
        raise ValueError("Job video thiếu version_id.")
    session_store.save_analysis_version(version_id, analysis=analysis, partial=[], status="done", job_id=job_id)
    document = session_store.get_document(job_data["document_id"]) or {}
    if document.get("source_hash") in (None, "", job_data.get("source_hash")):
        session_store.activate_analysis_version(job_data["document_id"], version_id)
    session_store.update_video_source(source["source_id"], status="analyzed", error="")
    update_job(
        job_id, "done", result=analysis, partial_result=[], stage="done",
        checkpoint={"completed": len(segments), "total": len(segments), "failed": len(failures)},
    )
    _dispatch_notion_for_video(job_data, source, segments, analysis)


def _run_video_visual(job_id: str, job_data: dict, payload: dict) -> None:
    source = session_store.get_video_source(job_data.get("source_id"))
    segment_id = str(payload.get("segment_id") or "")
    segment = next(
        (row for row in session_store.list_video_segments(job_data.get("source_id")) if row.get("segment_id") == segment_id),
        None,
    )
    if not source or not segment:
        raise ValueError("Không tìm thấy đoạn video cần phân tích hình ảnh.")
    update_job(job_id, "running", stage="visual_context", checkpoint={"segment_id": segment_id})
    visual, visual_usage = analyze_video_visual_segment(source, segment)
    analysis_row = _mapping(segment.get("analysis"))
    analysis_row["visual_context_detail"] = visual
    usage = _mapping(segment.get("usage"))
    usage["visual_context_usage"] = visual_usage
    session_store.update_video_segment(segment_id, analysis=analysis_row, usage=usage, status="done", error="")
    segments = session_store.list_video_segments(source["source_id"])
    analysis = build_video_analysis(
        {**source, "video_cues": session_store.list_video_cues(source["source_id"])}, segments
    )
    version_id = job_data.get("version_id")
    if version_id:
        session_store.save_analysis_version(version_id, analysis=analysis, status="done", job_id=job_id)
    update_job(job_id, "done", result={"job_kind": "video_visual", "segment_id": segment_id}, stage="done")
    if version_id:
        _dispatch_notion_for_video(job_data, source, segments, analysis)


def run_job(job_id: str, text: str, lang: str):
    try:
        update_job(job_id, "running")
        persist_main_result = False
        
        # Safe fallback: read from DB to avoid CLI character limits on long text
        job_data = get_job(job_id)
        if job_data:
            text = job_data["input_text"]
            lang = job_data["lang"]

        if job_data and job_data.get("job_kind") == "video_ingest":
            _run_video_ingest(job_id, job_data, json.loads(text or "{}"))
            return
        if job_data and job_data.get("job_kind") == "video_transcribe":
            _run_video_transcribe(job_id, job_data, json.loads(text or "{}"))
            return
        if job_data and job_data.get("job_kind") == "video_translate":
            _run_video_translate(job_id, job_data, json.loads(text or "{}"))
            return
        if job_data and job_data.get("job_kind") == "video_cue_recheck":
            _run_video_cue_recheck(job_id, job_data, json.loads(text or "{}"))
            return
        if job_data and job_data.get("job_kind") in {"video_analysis", "video_deep_analysis"}:
            payload = json.loads(text or "{}")
            if not isinstance(payload, dict):
                raise ValueError("Video job payload is invalid.")
            _run_video_analysis(job_id, job_data, payload)
            return
        if job_data and job_data.get("job_kind") == "video_visual":
            _run_video_visual(job_id, job_data, json.loads(text or "{}"))
            return

        if lang.startswith("guidance_"):
            data = json.loads(text)
            analysis_lang = "japanese" if lang == "guidance_ja" else "english"
            result = analyze_guidance_job(
                data.get("catalog", []),
                data.get("page_text", ""),
                analysis_lang,
                page_index=int(data.get("page_index", 0)),
                model_name=data.get("model_name"),
                reasoning_effort=data.get("reasoning_effort", "standard"),
            )
        elif lang.startswith("sentence_"):
            data = json.loads(text)
            analysis_lang = "japanese" if lang == "sentence_ja" else "english"
            result = analyze_manual_sentence(
                data["sentence"],
                data.get("page_text", ""),
                analysis_lang,
                model_name=data.get("model_name"),
                reasoning_effort=data.get("reasoning_effort", "standard"),
            )
        elif lang.startswith("pdf_"):
            persist_main_result = True
            # PDF/Image page analysis
            pages_data = json.loads(text)
            pages = pages_data.get("pages", [])
            model_name = pages_data.get("model_name")
            reasoning_effort = pages_data.get("reasoning_effort", "standard")
            auto_sentence_deep_dive = bool(pages_data.get("auto_sentence_deep_dive", True))
            auto_translation_guidance = bool(pages_data.get("auto_translation_guidance", True))
            analysis_mode = str(pages_data.get("analysis_mode") or "full_analysis")
            analysis_lang = "japanese" if lang == "pdf_ja" else "english"
            partial_results = list(job_data.get("partial_result") or []) if job_data else []

            def _persist_page(page_result: dict) -> None:
                page_index = int(page_result.get("page_index", 0))
                partial_results[:] = [
                    page for page in partial_results
                    if int(page.get("page_index", 0)) != page_index
                ]
                partial_results.append(page_result)
                partial_results.sort(key=lambda page: int(page.get("page_index", 0)))
                update_job(job_id, "running", partial_result=partial_results)
                if job_data and job_data.get("version_id"):
                    session_store.save_analysis_version(
                        job_data["version_id"], partial=partial_results, status="running", job_id=job_id
                    )

            result = run_page_analyses(
                pages,
                analysis_language=analysis_lang,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                auto_sentence_deep_dive=auto_sentence_deep_dive,
                auto_translation_guidance=auto_translation_guidance,
                analysis_mode=analysis_mode,
                page_done_callback=_persist_page,
            )
        else:
            persist_main_result = True
            # Fallback direct text analysis
            try:
                data = json.loads(text)
                raw_text = data.get("text", text)
                model_name = data.get("model_name")
                reasoning_effort = data.get("reasoning_effort", "standard")
            except Exception:
                raw_text = text
                model_name = None
                reasoning_effort = "standard"
            analysis_lang = "japanese" if lang in ("ja", "japanese") else "english"
            result = run_analysis(
                raw_text,
                [],
                analysis_language=analysis_lang,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
            )
            
        update_job(job_id, "done", result=result, partial_result=[])
        if job_data and job_data.get("document_id") and job_data.get("version_id") and (
            lang.startswith("guidance_") or lang.startswith("sentence_")
        ):
            version = session_store.get_analysis_version(job_data["version_id"]) or {}
            base_analysis = version.get("analysis")
            if lang.startswith("guidance_"):
                merged, changed = merge_guidance_job(base_analysis, result, job_id)
            else:
                merged, changed = merge_manual_breakdown(base_analysis, result, job_id)
            if changed:
                session_store.save_analysis_version(job_data["version_id"], analysis=merged)
                document = session_store.get_document(job_data["document_id"]) or {}
                if (
                    document.get("active_version_id") == job_data["version_id"]
                    and document.get("source_hash") == job_data.get("source_hash")
                    and notion_connection_state()["configured"]
                ):
                    settings = session_store.load_settings(job_data["session_id"])
                    if settings.get("auto_notion_sync", True):
                        notion_run = enqueue_analysis_sync(
                            job_data["session_id"], session_store.load_document_items(job_data["document_id"]), merged,
                            billing_tier=settings.get("billing_tier", "free"),
                            usd_to_jpy=float(settings.get("usd_to_jpy", 155)), document_id=job_data["document_id"],
                        )
                        if session_store.dispatch_notion_sync_run(notion_run["run_id"]):
                            subprocess.Popen(
                                [sys.executable, str(Path(__file__).resolve().parent / "notion_worker.py"), notion_run["run_id"]],
                                stdout=subprocess.DEVNULL,
                                stderr=open(str(Path(__file__).resolve().parent / "notion_worker_error.log"), "a"),
                                cwd=str(Path(__file__).resolve().parent),
                            )
            return
        if persist_main_result and job_data and job_data.get("session_id"):
            session_id = job_data["session_id"]
            document_id = job_data.get("document_id")
            version_id = job_data.get("version_id")
            items = (
                session_store.load_document_items(document_id)
                if document_id else session_store.load_image_items(session_id)
            )
            current_hash = items_source_hash(items)
            if version_id:
                # Always retain the historical result. It becomes the visible
                # version only when the document still has exactly this source.
                session_store.save_analysis_version(version_id, result, [], status="done", job_id=job_id)
                if not job_data.get("source_hash") or current_hash == job_data.get("source_hash"):
                    session_store.activate_analysis_version(document_id, version_id)
                    settings = session_store.load_settings(session_id)
                    if settings.get("auto_notion_sync", True) and notion_connection_state()["configured"]:
                        notion_run = enqueue_analysis_sync(
                            session_id, items, result,
                            billing_tier=settings.get("billing_tier", "free"),
                            usd_to_jpy=float(settings.get("usd_to_jpy", 155)),
                            document_id=document_id,
                        )
                        if session_store.dispatch_notion_sync_run(notion_run["run_id"]):
                            subprocess.Popen(
                                [sys.executable, str(Path(__file__).resolve().parent / "notion_worker.py"), notion_run["run_id"]],
                                stdout=subprocess.DEVNULL,
                                stderr=open(str(Path(__file__).resolve().parent / "notion_worker_error.log"), "a"),
                                cwd=str(Path(__file__).resolve().parent),
                            )
                return
            if not job_data.get("source_hash") or current_hash == job_data.get("source_hash"):
                settings = session_store.load_settings(session_id)
                selected_mode = str(settings.get("analysis_mode") or "full_analysis")
                result_mode = str(result.get("analysis_mode") or "full_analysis")
                if selected_mode != result_mode:
                    return
                session_store.save_analysis(session_id, result, [])
                if settings.get("auto_notion_sync", True) and notion_connection_state()["configured"]:
                    notion_run = enqueue_analysis_sync(
                        session_id,
                        items,
                        result,
                        billing_tier=settings.get("billing_tier", "free"),
                        usd_to_jpy=float(settings.get("usd_to_jpy", 155)),
                    )
                    if session_store.dispatch_notion_sync_run(notion_run["run_id"]):
                        subprocess.Popen(
                            [sys.executable, str(Path(__file__).resolve().parent / "notion_worker.py"), notion_run["run_id"]],
                            stdout=subprocess.DEVNULL,
                            stderr=open(str(Path(__file__).resolve().parent / "notion_worker_error.log"), "a"),
                            cwd=str(Path(__file__).resolve().parent),
                        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            failed_job = get_job(job_id) or {}
            if str(failed_job.get("job_kind") or "").startswith("video_") and failed_job.get("source_id"):
                session_store.update_video_source(failed_job["source_id"], status="failed", error=str(exc))
        except Exception:
            pass
        update_job(job_id, "failed", error=str(exc))


if __name__ == "__main__":
    job_id = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else ""
    lang = sys.argv[3] if len(sys.argv) > 3 else "ja"
    run_job(job_id, text, lang)
