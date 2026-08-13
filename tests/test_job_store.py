"""Unit tests for SQLite-backed job_store and worker."""

import pytest
import time
from modules.job_store import init_db, create_job, update_job, get_job, DB_PATH
from modules import job_store
import worker


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "DB_PATH", tmp_path / "jobs.db")


def test_job_lifecycle():
    # Make sure DB is initialized
    init_db()
    
    # Create job
    text = "Hello World"
    lang = "en"
    job_id = create_job(text, lang, session_id="abc123")
    assert job_id is not None
    
    # Get job and check pending status
    job = get_job(job_id)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["status"] == "pending"
    assert job["input_text"] == text
    assert job["lang"] == lang
    assert job["session_id"] == "abc123"
    
    # Update status to running
    update_job(job_id, "running")
    job = get_job(job_id)
    assert job["status"] == "running"
    
    # Update status to done with dummy result
    dummy_result = {"summary": "This is a summary", "vocabulary_all": []}
    update_job(job_id, "done", result=dummy_result)
    job = get_job(job_id)
    assert job["status"] == "done"
    assert job["result"] == dummy_result
    
    # Update status to failed with error message
    update_job(job_id, "failed", error="An error occurred")
    job = get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "An error occurred"


def test_partial_results_survive_running_updates():
    job_id = create_job("{}", "pdf_ja", session_id="abc123")
    partial = [{"page_index": 1, "summary": "Trang 1"}]
    update_job(job_id, "running", partial_result=partial)

    job = get_job(job_id)
    assert job["partial_result"] == partial
    assert job["result"] is None


def test_video_cue_recheck_rejects_stale_transcript(monkeypatch):
    monkeypatch.setattr(
        worker.session_store, "get_video_source",
        lambda source_id: {"source_id": source_id, "transcript_hash": "new-hash"},
    )
    with pytest.raises(ValueError, match="đã thay đổi"):
        worker._run_video_cue_recheck(
            "job", {"source_id": "source", "source_hash": "old-hash"},
            {"cue_id": "cue", "revision": 0},
        )


def test_auto_verify_can_restore_uncovered_speech(monkeypatch):
    monkeypatch.setattr(
        worker, "_transcribe_cue_proposal",
        lambda source, cue: (
            {
                "source_text": "Missing sentence.", "start_seconds": 10,
                "end_seconds": 12, "language": "english", "speaker": "",
                "confidence": "high", "uncertainty_reason": "",
            },
            {"model_used": "gemini-3.6-flash", "input_tokens": 10, "output_tokens": 5},
        ),
    )
    usage = {
        "runs": [{
            "run_id": "primary", "usage": {
                "coverage_gaps": [{"start": 10, "end": 12}]
            }
        }]
    }
    cues, updated = worker._auto_verify_transcript(
        {"source_id": "source", "duration_seconds": 20}, [], usage
    )
    assert cues[0]["source_text"] == "Missing sentence."
    assert cues[0]["verification_status"] == "verified_auto"
    assert [cue["start_seconds"] for cue in cues] == sorted(
        cue["start_seconds"] for cue in cues
    )
    assert updated["runs"][-1]["stage"] == "transcript_verification"


def test_video_worker_ignores_legacy_string_analysis_values():
    assert worker._has_sentence_breakdown({"analysis": "legacy raw response"}) is False
    assert worker._has_sentence_breakdown({"analysis": {"sentence_breakdown": {"original": "Hello"}}}) is True
