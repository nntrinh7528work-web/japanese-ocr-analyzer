"""Unit tests for SQLite-backed job_store and worker."""

import pytest
import time
from modules.job_store import init_db, create_job, update_job, get_job, DB_PATH
import worker


def test_job_lifecycle():
    # Make sure DB is initialized
    init_db()
    
    # Create job
    text = "Hello World"
    lang = "en"
    job_id = create_job(text, lang)
    assert job_id is not None
    
    # Get job and check pending status
    job = get_job(job_id)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["status"] == "pending"
    assert job["input_text"] == text
    assert job["lang"] == lang
    
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
