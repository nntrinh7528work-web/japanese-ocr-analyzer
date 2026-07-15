"""Standalone worker — run analysis in isolated process, save to DB."""

import sys
import json
from modules.job_store import update_job, get_job
from modules.text_analyzer import run_analysis, run_page_analyses


def run_job(job_id: str, text: str, lang: str):
    try:
        update_job(job_id, "running")
        
        # Safe fallback: read from DB to avoid CLI character limits on long text
        job_data = get_job(job_id)
        if job_data:
            text = job_data["input_text"]
            lang = job_data["lang"]
            
        from config import AI_PIPELINE_ENABLED
        use_pipeline = AI_PIPELINE_ENABLED or lang.startswith("ai_")

        if lang.startswith("pdf_") or lang.startswith("ai_"):
            # PDF/Image page analysis
            pages_data = json.loads(text)
            pages = pages_data.get("pages", [])
            analysis_lang = "japanese" if lang in ("pdf_ja", "ai_ja") else "english"
            if use_pipeline:
                from modules.analysis_pipeline import run_page_analyses_pipeline
                result = run_page_analyses_pipeline(pages, analysis_language=analysis_lang)
            else:
                result = run_page_analyses(pages, analysis_language=analysis_lang)

        else:
            # Fallback direct text analysis
            analysis_lang = "japanese" if lang in ("ja", "japanese", "ai_ja") else "english"
            if use_pipeline:
                from modules.analysis_pipeline import run_verified_analysis, adapt_for_ui
                pipeline_lang = "ja" if analysis_lang == "japanese" else "en"
                pipeline_result = run_verified_analysis(text, pipeline_lang)
                adapted = adapt_for_ui(pipeline_result["analysis"], text, analysis_lang)
                adapted["_pipeline_result"] = {
                    "review": pipeline_result["review"],
                    "quality_status": pipeline_result["quality_status"],
                    "warnings": pipeline_result["warnings"],
                }
                result = adapted
            else:
                result = run_analysis(text, [], analysis_language=analysis_lang)
            
        update_job(job_id, "done", result=result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        update_job(job_id, "failed", error=str(exc))


if __name__ == "__main__":
    job_id = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else ""
    lang = sys.argv[3] if len(sys.argv) > 3 else "ja"
    run_job(job_id, text, lang)
