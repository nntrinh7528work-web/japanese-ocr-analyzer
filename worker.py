"""Standalone worker — run analysis in isolated process, save to DB."""

import sys
import json
import subprocess
from pathlib import Path
from modules.job_store import update_job, get_job
from modules import session_store
from modules.job_workflow import items_source_hash
from modules.notion_sync import enqueue_analysis_sync, notion_connection_state
from modules.text_analyzer import run_analysis, run_page_analyses
from modules.sentence_analyzer import analyze_manual_sentence
from modules.translation_guidance import analyze_guidance_job


def run_job(job_id: str, text: str, lang: str):
    try:
        update_job(job_id, "running")
        persist_main_result = False
        
        # Safe fallback: read from DB to avoid CLI character limits on long text
        job_data = get_job(job_id)
        if job_data:
            text = job_data["input_text"]
            lang = job_data["lang"]

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
        if persist_main_result and job_data and job_data.get("session_id"):
            session_id = job_data["session_id"]
            items = session_store.load_image_items(session_id)
            current_hash = items_source_hash(items)
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
        update_job(job_id, "failed", error=str(exc))


if __name__ == "__main__":
    job_id = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else ""
    lang = sys.argv[3] if len(sys.argv) > 3 else "ja"
    run_job(job_id, text, lang)
