from modules.job_workflow import items_source_hash, sync_job_state


def test_new_completed_job_replaces_old_analysis():
    state = {
        "analysis": {"summary": "old"},
        "partial_page_analyses": [],
        "applied_job_id": "old-job",
    }
    job = {
        "status": "done",
        "session_id": "abc123",
        "result": {"summary": "new"},
        "partial_result": [],
    }

    status, changed = sync_job_state(state, "new-job", job, "abc123")

    assert status == "done"
    assert changed is True
    assert state["analysis"] == {"summary": "new"}
    assert state["applied_job_id"] == "new-job"


def test_foreign_job_cannot_modify_session_state():
    state = {"analysis": {"summary": "mine"}}
    job = {"status": "done", "session_id": "other", "result": {"summary": "foreign"}}

    status, changed = sync_job_state(state, "job", job, "mine")

    assert status == "foreign"
    assert changed is False
    assert state["analysis"] == {"summary": "mine"}


def test_stale_job_result_is_rejected_after_ocr_edit():
    old_items = [{"id": "1", "name": "p1", "edited_text": "old", "ocr_result": {}}]
    new_items = [{"id": "1", "name": "p1", "edited_text": "new", "ocr_result": {}}]
    state = {"analysis": None}
    job = {
        "status": "done",
        "session_id": "mine",
        "source_hash": items_source_hash(old_items),
        "result": {"summary": "old input"},
    }

    status, changed = sync_job_state(
        state, "job", job, "mine", current_source_hash=items_source_hash(new_items)
    )

    assert status == "stale"
    assert changed is False
    assert state["analysis"] is None


def test_stale_job_result_is_rejected_after_analysis_mode_change():
    state = {"analysis": None}
    job = {
        "status": "done",
        "session_id": "mine",
        "result": {"summary": "old mode", "analysis_mode": "full_analysis"},
    }

    status, changed = sync_job_state(
        state,
        "job",
        job,
        "mine",
        current_analysis_mode="sentence_guidance",
    )

    assert status == "stale"
    assert changed is False
    assert state["analysis"] is None


def test_sentence_job_merges_without_replacing_main_analysis_or_double_counting():
    state = {
        "analysis": {
            "summary": "main stays",
            "page_analyses": [
                {
                    "page_index": 1,
                    "sentence_catalog": [{"sentence_id": "p1-s1", "analyzed": False}],
                    "sentence_breakdowns": [],
                    "sentence_analysis_usage": {},
                }
            ],
        }
    }
    job = {
        "status": "done",
        "lang": "sentence_en",
        "session_id": "mine",
        "result": {
            "job_kind": "sentence_deep_dive",
            "page_index": 1,
            "sentence_id": "p1-s1",
            "breakdown": {"sentence_id": "p1-s1", "ordinal": 1},
            "usage": {"input_tokens": 4, "output_tokens": 6},
        },
    }

    status, changed = sync_job_state(state, "sentence-job", job, "mine")
    _status_again, changed_again = sync_job_state(state, "sentence-job", job, "mine")

    assert status == "done"
    assert changed is True
    assert changed_again is False
    assert state["analysis"]["summary"] == "main stays"
    assert state["analysis"]["sentence_analysis_usage"]["output_tokens"] == 6


def test_guidance_job_merges_without_replacing_main_analysis():
    state = {
        "analysis": {
            "summary": "main",
            "analysis_language": "english",
            "page_analyses": [{"page_index": 1, "translation_guidance": [], "vocabulary_all": []}],
        }
    }
    job = {
        "status": "done",
        "lang": "guidance_en",
        "session_id": "mine",
        "result": {
            "job_kind": "translation_guidance",
            "page_index": 1,
            "analysis_language": "english",
            "model_used": "gemini-test",
            "batch_results": [
                {
                    "batch_index": 1,
                    "rows": [{"sentence_id": "p1-s1", "ordinal": 1, "original": "Text.", "translations": {"natural": "Dịch."}}],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                    "error": None,
                }
            ],
        },
    }

    status, changed = sync_job_state(state, "guide-job", job, "mine")

    assert status == "done"
    assert changed is True
    assert state["analysis"]["summary"] == "main"
    assert state["analysis"]["translation_guidance_usage"]["output_tokens"] == 2
