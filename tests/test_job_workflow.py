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
