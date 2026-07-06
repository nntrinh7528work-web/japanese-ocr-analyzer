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
            
        if lang.startswith("pdf_"):
            # PDF/Image page analysis
            pages_data = json.loads(text)
            pages = pages_data.get("pages", [])
            analysis_lang = "japanese" if lang == "pdf_ja" else "english"
            result = run_page_analyses(pages, analysis_language=analysis_lang)

        else:
            # Fallback direct text analysis
            analysis_lang = "japanese" if lang in ("ja", "japanese") else "english"
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
