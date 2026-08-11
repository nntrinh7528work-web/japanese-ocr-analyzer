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
from modules.document_language import refresh_document_languages

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
from app_pages.video_page import render_video_tab

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
    "active_document_id": None,
    "selected_version_id": None,
    "working_version_id": None,
    "loaded_document_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Session persistence: restore or create ──────────────────────────────
session_store.cleanup_old_sessions(max_age_hours=24 * 30)
query_sid = st.query_params.get("session", "").strip()

if not st.session_state.session_restored:
    if query_sid and session_store.session_exists(query_sid):
        st.session_state.session_id = query_sid
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


def _load_document(document_id: str) -> None:
    """Load one independent document/version into the existing page widgets."""
    workspace = session_store.get_document_workspace(document_id)
    if not workspace:
        return
    active = workspace.get("active_version") or {}
    st.session_state.active_document_id = document_id
    st.session_state.loaded_document_id = document_id
    st.session_state.image_items = workspace["items"]
    st.session_state.analysis = active.get("analysis")
    st.session_state.partial_page_analyses = active.get("partial") or []
    st.session_state.selected_version_id = active.get("version_id")


documents = session_store.migrate_legacy_session_to_documents(st.session_state.session_id)
# AppTest, hot reload, and very old in-memory tabs can still arrive with image
# state before it has reached SQLite. Treat that state as a one-time legacy
# source rather than silently replacing it with an empty document.
if (
    st.session_state.image_items
    and len(documents) == 1
    and not session_store.load_document_items(documents[0]["document_id"])
):
    seed_document = documents[0]
    session_store.save_document_items(seed_document["document_id"], st.session_state.image_items)
    if st.session_state.analysis is not None:
        seed_language = str(st.session_state.analysis.get("analysis_language") or "japanese")
        seeded_version = session_store.create_analysis_version(
            seed_document["document_id"],
            items_source_hash(st.session_state.image_items),
            seed_language,
            str(st.session_state.analysis.get("analysis_mode") or "full_analysis"),
            st.session_state.analysis.get("model_used"), status="done",
            source_items=st.session_state.image_items,
        )
        session_store.save_analysis_version(
            seeded_version["version_id"], st.session_state.analysis,
            st.session_state.partial_page_analyses, status="done",
        )
        session_store.activate_analysis_version(seed_document["document_id"], seeded_version["version_id"])
    documents = session_store.list_documents(st.session_state.session_id)
requested_document_id = str(st.query_params.get("document") or "")
document_ids = {row["document_id"] for row in documents}
target_document_id = (
    requested_document_id if requested_document_id in document_ids
    else st.session_state.active_document_id if st.session_state.active_document_id in document_ids
    else documents[0]["document_id"]
)
if st.session_state.loaded_document_id != target_document_id:
    _load_document(target_document_id)
if st.query_params.get("document") != target_document_id:
    st.query_params["document"] = target_document_id


def _persist_items() -> None:
    """Save images/OCR only to the currently open document."""
    sid = st.session_state.session_id
    document_id = st.session_state.active_document_id
    if sid and document_id:
        document = session_store.get_document(document_id) or {}
        primary, source = refresh_document_languages(
            st.session_state.image_items, str(document.get("language") or "unknown")
        )
        if document.get("language") == "unknown" and primary != "unknown":
            session_store.update_document_language(document_id, primary, source)
        session_store.save_document_items(document_id, st.session_state.image_items)
        session_store.update_session_timestamp(sid)


def _persist_analysis() -> None:
    """Persist only the selected/working version, never the whole session."""
    sid = st.session_state.session_id
    selected = st.session_state.selected_version_id
    working = st.session_state.working_version_id
    if sid and selected and st.session_state.analysis is not None:
        session_store.save_analysis_version(selected, analysis=st.session_state.analysis)
    if sid and working:
        session_store.save_analysis_version(working, partial=st.session_state.partial_page_analyses)
    if sid:
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
    if job and job.get("document_id"):
        if job.get("session_id") != st.session_state.session_id:
            job_status = "foreign"
        else:
            job_status = str(job.get("status") or "pending")
            if job.get("document_id") == st.session_state.active_document_id:
                if job.get("partial_result"):
                    st.session_state.partial_page_analyses = job["partial_result"]
                    st.session_state.working_version_id = job.get("version_id")
                if job_status == "done":
                    _load_document(st.session_state.active_document_id)
            job_changed = False
    elif job:
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
            if job.get("job_kind") == "video_ingest":
                st.success("Đã hoàn tất bước lấy transcript. Hãy kiểm tra dự toán trước khi phân tích.")
            else:
                st.success("Phân tích đã hoàn tất!")
        elif job_status == "running":
            stage = str(job.get("stage") or "")
            video_label = "Đang xử lý video" if job.get("job_kind", "").startswith("video_") else "Đang phân tích"
            st.warning(f"{video_label} trong nền ({stage or 'đang chạy'}), vui lòng tải lại sau vài giây...")
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
    """Mark a document changed while retaining its historical versions."""
    _persist_items()


def analysis_pages(items: list[dict]) -> list[dict]:
    """Prepare page dictionaries for text analysis."""
    pages = []
    for index, item in enumerate(items, 1):
        if item.get("mismatch_status") == "mismatch":
            continue
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


def add_sources(sources: list[tuple[str, bytes]], destination: str = "current") -> bool:
    """Add files to this document or create a fresh document intentionally."""
    if destination == "new":
        title = Path(sources[0][0]).stem if sources else "Bài mới"
        document = session_store.create_document(st.session_state.session_id, title)
        st.session_state.active_document_id = document["document_id"]
        st.session_state.loaded_document_id = document["document_id"]
        st.session_state.selected_version_id = None
        st.session_state.working_version_id = None
        st.session_state.analysis = None
        st.session_state.partial_page_analyses = []
        st.session_state.image_items = []
        st.query_params["document"] = document["document_id"]
    items, added, errors = add_upload_items(st.session_state.image_items, sources)
    st.session_state.image_items = items
    st.session_state.upload_messages = []
    st.session_state.upload_errors = errors
    if added:
        st.session_state.upload_messages.append(f"Đã thêm {len(added)} ảnh/trang PDF.")
        _persist_items()
    return bool(added)


def remove_image(item_id: str) -> None:
    """Remove image item by ID."""
    st.session_state.image_items = [item for item in st.session_state.image_items if item["id"] != item_id]
    _persist_items()


def move_image_to_new_document(item_id: str) -> None:
    document = session_store.move_document_item_to_new_document(
        st.session_state.active_document_id, item_id, st.session_state.session_id
    )
    if document:
        st.session_state.active_document_id = document["document_id"]
        st.session_state.loaded_document_id = None
        st.query_params["document"] = document["document_id"]


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


def create_analysis_version_for_active(
    analysis_mode: str, language: str, model_name: str | None,
) -> dict:
    """Reserve an immutable result version before sending a background job."""
    document_id = st.session_state.active_document_id
    source_hash = items_source_hash(st.session_state.image_items)
    version = session_store.create_analysis_version(
        document_id,
        source_hash,
        language,
        analysis_mode,
        model_name,
        status="running",
        source_items=st.session_state.image_items,
    )
    st.session_state.working_version_id = version["version_id"]
    st.session_state.partial_page_analyses = []
    return version


# ── Title & Branding ──
render_branded_header()

# ── Sidebar & Theme Injection ──
config = render_sidebar(
    st.session_state.image_items,
    clear_analysis_fn=clear_analysis,
    persist_items_fn=_persist_items,
    persist_analysis_fn=_persist_analysis,
)
active_document = session_store.get_document(st.session_state.active_document_id) or {}
config["analysis_language"] = (
    active_document.get("language") if active_document.get("language") in {"japanese", "english"} else "japanese"
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


def _document_label(document: dict) -> str:
    language = {"japanese": "JP", "english": "EN", "unknown": "?"}.get(document.get("language"), "?")
    source_label = "1 video" if document.get("document_type") == "video" else f"{document.get('image_count', 0)} ảnh"
    return f"[{language}] {document.get('title') or 'Bài mới'} · {source_label} · {document.get('version_count', 0)} phiên bản"


def render_document_library() -> None:
    """Small, mobile-friendly document switcher above the OCR workspace."""
    docs = session_store.list_documents(st.session_state.session_id)
    filters = {"Tất cả": None, "Video": "video", "Tiếng Nhật": "japanese", "Tiếng Anh": "english", "Đang xử lý": "running", "Đã phân tích": "analyzed"}
    filter_label = st.selectbox("Lọc thư viện bài", list(filters), key="document_library_filter")
    wanted = filters[filter_label]
    visible = [
        doc for doc in docs
        if wanted is None or doc.get("document_type") == wanted or doc.get("language") == wanted or doc.get("status") == wanted
    ]
    with st.expander("📚 Thư viện bài", expanded=True):
        left, right = st.columns([4, 1])
        options = { _document_label(doc): doc["document_id"] for doc in visible }
        current = st.session_state.active_document_id
        option_labels = list(options) or [_document_label(active_document)]
        current_index = next((i for i, label in enumerate(option_labels) if options.get(label) == current), 0)
        selected_label = left.selectbox("Bài đang mở", option_labels, index=current_index, key="document_library_picker")
        if right.button("＋ Bài mới", use_container_width=True):
            document = session_store.create_document(st.session_state.session_id)
            st.session_state.active_document_id = document["document_id"]
            st.session_state.loaded_document_id = None
            st.query_params["document"] = document["document_id"]
            st.rerun()
        selected_id = options.get(selected_label, current)
        if selected_id and selected_id != current:
            st.session_state.active_document_id = selected_id
            st.session_state.loaded_document_id = None
            st.query_params["document"] = selected_id
            st.rerun()

        document = session_store.get_document(st.session_state.active_document_id) or {}
        rename_col, language_col, save_col = st.columns([3, 2, 1])
        title = rename_col.text_input("Tên bài", value=document.get("title") or "Bài mới", key=f"document_title_{document.get('document_id')}")
        language_options = {"Tự nhận diện": "unknown", "Tiếng Nhật": "japanese", "Tiếng Anh": "english"}
        selected_language = language_col.selectbox(
            "Ngôn ngữ bài", list(language_options),
            index=list(language_options.values()).index(document.get("language", "unknown")),
            key=f"document_language_{document.get('document_id')}",
        )
        if save_col.button("Lưu", key=f"save_document_{document.get('document_id')}", use_container_width=True):
            session_store.rename_document(document["document_id"], title)
            session_store.update_document_language(document["document_id"], language_options[selected_language], "manual")
            _persist_items()
            st.rerun()

        versions = session_store.list_analysis_versions(document.get("document_id", ""))
        if versions:
            version_options = {
                f"Phiên bản {row['version_number']} · {row['status']} · {row['created_at'][:16]}": row["version_id"]
                for row in versions
            }
            selected_version = st.selectbox("Xem kết quả phiên bản", list(version_options), key=f"version_picker_{document['document_id']}")
            version_id = version_options[selected_version]
            if version_id != st.session_state.selected_version_id:
                version = session_store.get_analysis_version(version_id) or {}
                st.session_state.selected_version_id = version_id
                st.session_state.analysis = version.get("analysis")
                st.session_state.partial_page_analyses = version.get("partial") or []
        if document.get("active_version_id") and document.get("source_hash") != (session_store.get_analysis_version(document["active_version_id"]) or {}).get("source_hash"):
            st.warning("Đã có ảnh hoặc OCR mới. Kết quả phiên bản đang xem vẫn được giữ; hãy phân tích lại để tạo phiên bản mới.")


render_document_library()


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
    and active_document.get("document_type", "image") == "image"
    and active_document.get("active_version_id") == st.session_state.selected_version_id
    and active_document.get("source_hash") == items_source_hash(st.session_state.image_items)
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
        document_id=st.session_state.active_document_id,
        )
        _dispatch_notion_run(notion_run["run_id"])
    except Exception as exc:
        st.sidebar.warning(f"Chưa thể tạo hàng đợi Notion: {exc}")

if notion_state["configured"] and migration_status in {"complete", "partial", "not_needed"}:
    for due_run in session_store.list_due_notion_sync_runs(limit=2):
        _dispatch_notion_run(due_run["run_id"])

# ── Main Content Tabs ──
tab_ocr, tab_video, tab_dialogue = st.tabs(["📷 Phân tích từ Ảnh / PDF", "YouTube / Video", "💬 Luyện Hội Thoại"])

with tab_ocr:
    if active_document.get("document_type") == "video":
        st.info("Bài đang mở là video. Hãy tạo hoặc mở một bài ảnh/PDF trong Thư viện để tránh trộn hai nguồn.")
        if st.button("Tạo bài ảnh/PDF mới", use_container_width=True):
            document = session_store.create_document(st.session_state.session_id, "Bài ảnh/PDF mới")
            st.session_state.active_document_id = document["document_id"]
            st.session_state.loaded_document_id = None
            st.query_params["document"] = document["document_id"]
            st.rerun()
    else:
        render_ocr_tab(
            config=config,
            active_document=active_document,
            selected_version=session_store.get_analysis_version(st.session_state.selected_version_id) if st.session_state.selected_version_id else None,
            add_sources_fn=add_sources,
            remove_image_fn=remove_image,
            move_image_to_new_document_fn=move_image_to_new_document,
            clear_analysis_fn=clear_analysis,
            analysis_pages_fn=analysis_pages,
            run_item_ocr_fn=run_item_ocr,
            persist_items_fn=_persist_items,
            persist_analysis_fn=_persist_analysis,
            create_analysis_version_fn=create_analysis_version_for_active,
            text_analyzer_module=text_analyzer,
            worker_path=_WORKER_PATH,
            notion_worker_path=_NOTION_WORKER_PATH,
            project_dir=_PROJECT_DIR,
        )

with tab_video:
    render_video_tab(
        config=config,
        active_document=active_document,
        worker_path=_WORKER_PATH,
        project_dir=_PROJECT_DIR,
    )

with tab_dialogue:
    render_dialogue_tab(
        session_id=st.session_state.session_id,
        model_name=config.get("text_model_choice"),
        reasoning_effort=config.get("reasoning_effort", "standard"),
        billing_tier=config.get("billing_tier", "free"),
        usd_to_jpy=float(config.get("usd_to_jpy", 155.0)),
    )
