"""Streamlit entry point for Japanese/English OCR Analyzer."""

from __future__ import annotations
import importlib
from pathlib import Path
import subprocess
import sys
import streamlit as st

from modules.multi_image_workflow import add_upload_items
from modules import session_store
import modules.text_analyzer as text_analyzer
from modules.job_store import cleanup_old_jobs, get_job
from modules.job_workflow import items_source_hash, sync_job_state
from modules.notion_sync import enqueue_analysis_sync, notion_connection_state

from components.styles import inject_custom_css, render_branded_header
from components.sidebar import render_sidebar
from components.helpers import (
    COLUMN_LABELS,
    display_rows,
    render_example,
    render_important_vocabulary,
    render_grammar_points,
)
from app_pages.ocr_page import render_ocr_tab
from app_pages.dialogue_page import render_dialogue_tab

text_analyzer = importlib.reload(text_analyzer)

st.set_page_config(page_title="Japanese / English OCR Analyzer", page_icon="🔍", layout="wide")

# Session State Initialization
for key, default in {
    "image_items": [],
    "analysis": None,
    "partial_page_analyses": [],
    "upload_messages": [],
    "upload_errors": [],
    "uploader_version": 0,
    "camera_version": 0,
    "session_id": None,
    "session_restored": False,
    "current_job_id": None,
    "applied_job_id": None,
    "budget_jpy": 0.0,
    "spent_before_jpy": 0.0,
    "usd_to_jpy": 155.0,
    "recent_topics": [],
    "dialogue_result": None,
    "dialogue_quiz": None,
    "revealed_turns": {},
    "dialogue_history_id": None,
    "dialogue_quiz_results": {},
    "dark_mode": False,
    "auto_sentence_deep_dive": False,
    "auto_translation_guidance": True,
    "analysis_mode": "full_analysis",
    "auto_notion_sync": True,
    "billing_tier": "free",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Session persistence: restore or create ──────────────────────────────
session_store.cleanup_old_sessions(max_age_hours=24 * 30)
query_sid = st.query_params.get("session", "").strip()

if not st.session_state.session_restored:
    if query_sid and session_store.session_exists(query_sid):
        st.session_state.session_id = query_sid
        saved_items = session_store.load_image_items(query_sid)
        if saved_items:
            st.session_state.image_items = saved_items
        saved_analysis, saved_partial = session_store.load_analysis(query_sid)
        if saved_analysis:
            st.session_state.analysis = saved_analysis
        if saved_partial:
            st.session_state.partial_page_analyses = saved_partial
        saved_settings = session_store.load_settings(query_sid)
        for key in (
            "budget_jpy",
            "spent_before_jpy",
            "usd_to_jpy",
            "auto_sentence_deep_dive",
            "auto_translation_guidance",
            "analysis_mode",
            "auto_notion_sync",
            "billing_tier",
        ):
            if key in saved_settings:
                st.session_state[key] = saved_settings[key]
        session_store.update_session_timestamp(query_sid)
    else:
        new_sid = session_store.generate_session_id()
        session_store.create_session(new_sid)
        st.session_state.session_id = new_sid
        st.query_params["session"] = new_sid
    st.session_state.session_restored = True

if st.session_state.session_id and st.query_params.get("session") != st.session_state.session_id:
    st.query_params["session"] = st.session_state.session_id


def _persist_items() -> None:
    """Save current image items to SQLite."""
    sid = st.session_state.session_id
    if sid:
        session_store.save_image_items(sid, st.session_state.image_items)
        session_store.update_session_timestamp(sid)


def _persist_analysis() -> None:
    """Save current analysis and partial results to SQLite."""
    sid = st.session_state.session_id
    if sid:
        session_store.save_analysis(
            sid,
            st.session_state.analysis,
            st.session_state.partial_page_analyses,
        )
        session_store.update_session_timestamp(sid)


_WORKER_PATH = str(Path(__file__).resolve().parent / "worker.py")
_NOTION_WORKER_PATH = str(Path(__file__).resolve().parent / "notion_worker.py")
_NOTION_MIGRATION_WORKER_PATH = str(
    Path(__file__).resolve().parent / "notion_migration_worker.py"
)
_PROJECT_DIR = str(Path(__file__).resolve().parent)
cleanup_old_jobs(max_age_hours=48)

# ── Check background job status if job_id is present ─────────────────────
job_id_from_url = st.query_params.get("job_id")
if job_id_from_url:
    job = get_job(job_id_from_url)
    job_status = None
    job_changed = False
    if job:
        job_status, job_changed = sync_job_state(
            st.session_state,
            job_id_from_url,
            job,
            st.session_state.session_id,
            current_source_hash=items_source_hash(st.session_state.image_items),
            current_analysis_mode=st.session_state.get("analysis_mode", "full_analysis"),
        )
    if job_status == "foreign":
        st.error("Job phân tích này không thuộc phiên làm việc hiện tại.")
    elif job_status == "stale":
        st.warning("Nội dung OCR đã thay đổi nên kết quả job cũ không được áp dụng.")
    elif job:
        job_partial = job.get("partial_result") or []
        if job_changed:
            _persist_analysis()
        if job_status == "done":
            st.success("Phân tích đã hoàn tất!")
        elif job_status == "running":
            st.warning("⏳ Đang phân tích trong nền, vui lòng tải lại trang sau vài giây...")
            st.button("🔄 Tải lại")
        elif job_status == "failed":
            st.error(f"Phân tích thất bại: {job['error']}")
            retry_label = "▶️ Tiếp tục các trang còn lại" if job_partial else "🔄 Thử lại từ đầu"
            if st.button(retry_label):
                del st.query_params["job_id"]
                st.rerun()
        else:
            st.info("⏳ Đang chờ xử lý...")
            st.button("🔄 Tải lại")


def clear_analysis() -> None:
    """Clear active analysis results."""
    st.session_state.analysis = None
    st.session_state.partial_page_analyses = []
    st.session_state.current_job_id = None
    st.session_state.applied_job_id = None
    if "job_id" in st.query_params:
        del st.query_params["job_id"]
    _persist_analysis()


def analysis_pages(items: list[dict]) -> list[dict]:
    """Prepare page dictionaries for text analysis."""
    pages = []
    for index, item in enumerate(items, 1):
        text = str(item.get("edited_text") or "").strip()
        if not text:
            continue
        result = item.get("ocr_result") or {}
        pages.append(
            {
                "page_index": index,
                "page_name": item.get("name") or f"Trang {index}",
                "text": text,
                "notes": result.get("ocr_notes", []),
            }
        )
    return pages


def add_sources(sources: list[tuple[str, bytes]]) -> bool:
    """Add uploaded image or PDF sources."""
    items, added, errors = add_upload_items(st.session_state.image_items, sources)
    st.session_state.image_items = items
    st.session_state.upload_messages = []
    st.session_state.upload_errors = errors
    if added:
        clear_analysis()
        st.session_state.upload_messages.append(f"Đã thêm {len(added)} ảnh/trang PDF.")
        _persist_items()
    return bool(added)


def remove_image(item_id: str) -> None:
    """Remove image item by ID."""
    st.session_state.image_items = [item for item in st.session_state.image_items if item["id"] != item_id]
    clear_analysis()
    _persist_items()


def run_item_ocr(item: dict, model_name: str | None = None) -> None:
    """Run OCR on a single image item."""
    from modules.ocr_engine import run_ocr
    item["ocr_error"] = None
    try:
        result = run_ocr(item["processed_image_bytes"], item["report"], model_name=model_name)
        item["ocr_result"] = result
        item["edited_text"] = result["clean_text"]
    except Exception as exc:
        item["ocr_error"] = str(exc)


# ── Title & Branding ──
render_branded_header()

# ── Sidebar & Theme Injection ──
config = render_sidebar(
    st.session_state.image_items,
    clear_analysis_fn=clear_analysis,
    persist_items_fn=_persist_items,
    persist_analysis_fn=_persist_analysis,
)
session_store.save_settings(
    st.session_state.session_id,
    {
        "budget_jpy": config["budget_jpy"],
        "spent_before_jpy": config["spent_before_jpy"],
        "usd_to_jpy": config["usd_to_jpy"],
        "auto_sentence_deep_dive": config["auto_sentence_deep_dive"],
        "auto_translation_guidance": config["auto_translation_guidance"],
        "analysis_mode": config.get(
            "analysis_mode", st.session_state.get("analysis_mode", "full_analysis")
        ),
        "auto_notion_sync": config["auto_notion_sync"],
        "billing_tier": config["billing_tier"],
    },
)
inject_custom_css(dark_mode=config.get("dark_mode", False))


def _dispatch_notion_run(run_id: str) -> bool:
    """Launch one already-persisted Notion run exactly once."""
    if not session_store.dispatch_notion_sync_run(run_id):
        return False
    try:
        subprocess.Popen(
            [sys.executable, _NOTION_WORKER_PATH, run_id],
            stdout=subprocess.DEVNULL,
            stderr=open(str(Path(_PROJECT_DIR) / "notion_worker_error.log"), "a"),
            cwd=_PROJECT_DIR,
        )
        return True
    except Exception as exc:
        session_store.finish_notion_sync_run(run_id, "retry", error=str(exc))
        return False


def _dispatch_notion_migration() -> bool:
    if st.session_state.get("_notion_v4_migration_dispatched"):
        return False
    st.session_state["_notion_v4_migration_dispatched"] = True
    try:
        subprocess.Popen(
            [sys.executable, _NOTION_MIGRATION_WORKER_PATH],
            stdout=subprocess.DEVNULL,
            stderr=open(str(Path(_PROJECT_DIR) / "notion_migration_error.log"), "a"),
            cwd=_PROJECT_DIR,
        )
        return True
    except Exception as exc:
        st.sidebar.warning(f"Chưa thể khởi động nâng cấp Notion: {exc}")
        return False


notion_state = notion_connection_state()
migration_status = str(
    (session_store.load_notion_workspace_config().get("migration_v4") or {}).get("status") or ""
)
if notion_state["configured"] and migration_status not in {"complete", "partial", "not_needed"}:
    _dispatch_notion_migration()

# A completed analysis is displayed immediately; Notion runs separately.
if (
    st.session_state.analysis
    and config.get("auto_notion_sync")
    and notion_state["configured"]
    and migration_status in {"complete", "partial", "not_needed"}
):
    try:
        notion_run = enqueue_analysis_sync(
            st.session_state.session_id,
            st.session_state.image_items,
            st.session_state.analysis,
            billing_tier=config.get("billing_tier", "free"),
            usd_to_jpy=config.get("usd_to_jpy", 155),
        )
        _dispatch_notion_run(notion_run["run_id"])
    except Exception as exc:
        st.sidebar.warning(f"Chưa thể tạo hàng đợi Notion: {exc}")

if notion_state["configured"] and migration_status in {"complete", "partial", "not_needed"}:
    for due_run in session_store.list_due_notion_sync_runs(limit=2):
        _dispatch_notion_run(due_run["run_id"])

# ── Main Content Tabs ──
tab_ocr, tab_dialogue = st.tabs(["📷 Phân tích từ Ảnh / PDF", "💬 Luyện Hội Thoại"])

with tab_ocr:
    render_ocr_tab(
        config=config,
        add_sources_fn=add_sources,
        remove_image_fn=remove_image,
        clear_analysis_fn=clear_analysis,
        analysis_pages_fn=analysis_pages,
        run_item_ocr_fn=run_item_ocr,
        persist_items_fn=_persist_items,
        persist_analysis_fn=_persist_analysis,
        text_analyzer_module=text_analyzer,
        worker_path=_WORKER_PATH,
        notion_worker_path=_NOTION_WORKER_PATH,
        project_dir=_PROJECT_DIR,
    )

with tab_dialogue:
    render_dialogue_tab(session_id=st.session_state.session_id)
