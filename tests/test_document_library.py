"""Coverage for independent documents, versions, and local language guards."""

from modules import session_store
from modules.document_language import refresh_document_languages
from modules.job_workflow import items_source_hash


def _item(item_id: str, name: str, text: str) -> dict:
    return {
        "id": item_id,
        "name": name,
        "original_image_bytes": b"source",
        "processed_image_bytes": b"processed",
        "report": {},
        "ocr_result": {"ocr_notes": []},
        "edited_text": text,
        "ocr_error": None,
    }


def test_legacy_session_migrates_to_a_version_without_reanalysis(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_DB_PATH", str(tmp_path / "sessions.db"))
    session_store.create_session("legacy")
    source = [_item("jp", "lesson.png", "これは日本語です。")]
    session_store.save_image_items("legacy", source)
    session_store.save_analysis("legacy", {"analysis_language": "japanese", "summary": "saved"})

    documents = session_store.migrate_legacy_session_to_documents("legacy")
    workspace = session_store.get_document_workspace(documents[0]["document_id"])

    assert len(documents) == 1
    assert workspace["items"][0]["edited_text"] == source[0]["edited_text"]
    assert workspace["active_version"]["version_number"] == 1
    assert workspace["active_version"]["analysis"]["summary"] == "saved"
    assert workspace["active_version"]["source_snapshot"][0]["id"] == "jp"
    assert session_store.migrate_legacy_session_to_documents("legacy")[0]["document_id"] == documents[0]["document_id"]


def test_documents_and_versions_do_not_mix_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_DB_PATH", str(tmp_path / "sessions.db"))
    session_store.create_session("many")
    jp = session_store.create_document("many", "Japanese", "japanese")
    en = session_store.create_document("many", "English", "english")
    jp_items = [_item("jp", "jp.png", "これは日本語です。")]
    en_items = [_item("en", "en.png", "This is English.")]
    session_store.save_document_items(jp["document_id"], jp_items)
    session_store.save_document_items(en["document_id"], en_items)

    first = session_store.create_analysis_version(
        jp["document_id"], items_source_hash(jp_items), "japanese", "full_analysis", source_items=jp_items
    )
    session_store.save_analysis_version(first["version_id"], {"summary": "JP"}, [], status="done")
    session_store.activate_analysis_version(jp["document_id"], first["version_id"])

    second = session_store.create_analysis_version(
        en["document_id"], items_source_hash(en_items), "english", "sentence_guidance", source_items=en_items
    )
    session_store.save_analysis_version(second["version_id"], {"summary": "EN"}, [], status="done")
    session_store.activate_analysis_version(en["document_id"], second["version_id"])

    assert session_store.get_document_workspace(jp["document_id"])["active_version"]["analysis"]["summary"] == "JP"
    assert session_store.get_document_workspace(en["document_id"])["active_version"]["analysis"]["summary"] == "EN"


def test_added_source_keeps_old_version_snapshot_and_detects_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_DB_PATH", str(tmp_path / "sessions.db"))
    session_store.create_session("versions")
    document = session_store.create_document("versions", "Japanese", "japanese")
    items = [_item("jp", "jp.png", "これは日本語です。")]
    refresh_document_languages(items, "japanese")
    session_store.save_document_items(document["document_id"], items)
    version = session_store.create_analysis_version(
        document["document_id"], items_source_hash(items), "japanese", "full_analysis", source_items=items
    )
    session_store.save_analysis_version(version["version_id"], {"summary": "before"}, [], status="done")
    session_store.activate_analysis_version(document["document_id"], version["version_id"])

    items.append(_item("en", "en.png", "Although this page is English, it is a new source."))
    refresh_document_languages(items, "japanese")
    session_store.save_document_items(document["document_id"], items)
    old = session_store.get_analysis_version(version["version_id"])
    current = session_store.get_document_workspace(document["document_id"])

    assert items[1]["mismatch_status"] == "mismatch"
    assert [row["id"] for row in old["source_snapshot"]] == ["jp"]
    assert current["active_version"]["analysis"]["summary"] == "before"
    assert current["source_hash"] != old["source_hash"]
