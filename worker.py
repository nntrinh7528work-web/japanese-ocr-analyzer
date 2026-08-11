"""Standalone worker — run analysis in isolated process, save to DB."""

import sys
import json
import subprocess
from pathlib import Path
from config import MAX_VIDEO_DURATION_SECONDS
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
    build_segment_batches,
    build_segments,
    build_video_analysis,
    clean_transcript,
    fetch_youtube_caption,
    normalize_transcript,
    transcript_hash,
    transcribe_with_gemini,
)


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


def _run_video_ingest(job_id: str, job_data: dict, payload: dict) -> None:
    source = session_store.get_video_source(job_data.get("source_id"))
    if not source:
        raise ValueError("Không tìm thấy nguồn video của job.")
    update_job(job_id, "running", stage="fetching_transcript", checkpoint={"window": 0})
    allow_gemini = bool(payload.get("allow_gemini", False))
    provider = ""
    usage: dict = {}
    try:
        if source.get("source_kind") == "youtube" and not allow_gemini:
            rows, provider = fetch_youtube_caption(
                str(source.get("video_id") or ""), str(payload.get("preferred_language") or "unknown")
            )
        elif allow_gemini:
            rows, usage = transcribe_with_gemini(source)
            provider = "gemini_video"
        else:
            session_store.update_video_source(source["source_id"], status="awaiting_ingest_confirmation")
            update_job(
                job_id, "done", stage="awaiting_ingest_confirmation",
                result={"job_kind": "video_ingest", "status": "awaiting_ingest_confirmation"},
            )
            return
    except Exception as exc:
        if source.get("source_kind") == "youtube" and not allow_gemini:
            session_store.update_video_source(
                source["source_id"], status="awaiting_ingest_confirmation", error=str(exc)
            )
            update_job(
                job_id, "done", stage="awaiting_ingest_confirmation",
                result={"job_kind": "video_ingest", "status": "awaiting_ingest_confirmation", "reason": str(exc)},
            )
            return
        raise

    raw_rows = normalize_transcript(rows)
    clean_rows, cleanup_warnings = clean_transcript(raw_rows)
    duration = float(
        source.get("duration_seconds") or usage.get("duration_seconds") or (raw_rows[-1]["end"] if raw_rows else 0)
    )
    selected_range = source.get("metadata") or {}
    has_selected_range = (
        selected_range.get("range_start") is not None and selected_range.get("range_end") is not None
        and float(selected_range.get("range_end") or 0) > float(selected_range.get("range_start") or 0)
    )
    current_transcript_hash = transcript_hash(raw_rows)
    source_metadata = {
        **(source.get("metadata") or {}),
        "visual_context": usage.get("visual_context") or [],
    }
    transcript_warnings = [*(usage.get("warnings") or []), *cleanup_warnings]
    if duration > MAX_VIDEO_DURATION_SECONDS and not has_selected_range:
        session_store.update_video_source(
            source["source_id"], duration_seconds=duration, raw_transcript=raw_rows,
            clean_transcript=[], transcript_warnings=transcript_warnings,
            metadata=source_metadata,
            transcript_provider=provider, transcript_hash=current_transcript_hash,
            ingest_usage=usage, status="awaiting_range_selection", error="",
        )
        session_store.update_document_source_hash(
            job_data["document_id"], current_transcript_hash, status="awaiting_range_selection"
        )
        update_job(
            job_id, "done", stage="awaiting_range_selection",
            result={
                "job_kind": "video_ingest", "status": "awaiting_range_selection",
                "source_id": source["source_id"], "duration_seconds": duration,
            },
        )
        return
    session_store.update_video_source(
        source["source_id"], duration_seconds=duration, raw_transcript=raw_rows,
        clean_transcript=clean_rows, transcript_warnings=transcript_warnings,
        metadata=source_metadata,
        transcript_provider=provider, transcript_hash=current_transcript_hash,
        ingest_usage=usage, status="segmenting", error="",
    )
    session_store.update_document_source_hash(
        job_data["document_id"], current_transcript_hash, status="awaiting_cost_confirmation"
    )
    segments = build_segments(clean_rows)
    session_store.replace_video_segments(source["source_id"], segments)
    source = session_store.get_video_source(source["source_id"]) or source
    estimate = build_cost_estimate(source, session_store.list_video_segments(source["source_id"]), payload.get("billing_tier", "free"))
    session_store.update_video_source(
        source["source_id"], cost_estimate=estimate, status="awaiting_cost_confirmation"
    )
    local_path = str(source.get("local_path") or "")
    if local_path and allow_gemini:
        try:
            Path(local_path).unlink(missing_ok=True)
            session_store.update_video_source(source["source_id"], local_path="")
        except OSError:
            pass
    update_job(
        job_id, "done", stage="awaiting_cost_confirmation",
        result={
            "job_kind": "video_ingest", "status": "awaiting_cost_confirmation",
            "source_id": source["source_id"], "segment_count": len(segments), "cost_estimate": estimate,
        },
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
    deep_count = sum(1 for row in segments if (row.get("analysis") or {}).get("sentence_breakdown"))
    failures = []
    pending = [row for row in segments if not (row.get("status") == "done" and row.get("analysis"))]
    batches = build_segment_batches(pending)
    completed_count = len(segments) - len(pending)

    def persist_result(segment: dict, result: dict, usage: dict) -> None:
        nonlocal deep_count, completed_count
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
            batch_usage["run_id"] = f"{job_id}:batch:{batch_index + 1}"
            for segment, result in zip(batch, results):
                persist_result(segment, result, dict(batch_usage))
        except Exception as batch_error:
            # A malformed result for one row must not discard neighboring segments.
            if len(batch) > 1:
                for segment in batch:
                    try:
                        rows, usage = analyze_video_segment_batch([segment])
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
    analysis = build_video_analysis(source, segments)
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
    analysis_row = dict(segment.get("analysis") or {})
    analysis_row["visual_context_detail"] = visual
    usage = dict(segment.get("usage") or {})
    usage["visual_context_usage"] = visual_usage
    session_store.update_video_segment(segment_id, analysis=analysis_row, usage=usage, status="done", error="")
    segments = session_store.list_video_segments(source["source_id"])
    analysis = build_video_analysis(source, segments)
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
        if job_data and job_data.get("job_kind") == "video_analysis":
            _run_video_analysis(job_id, job_data, json.loads(text or "{}"))
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
