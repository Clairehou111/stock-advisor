"""
Sid Sloth — Streamlit frontend.

Pages:
  - Login (shared password → JWT)
  - Chat (main chatbot interface)
  - Admin (admin only: trigger ingestion pipelines)
"""

import base64
import json
import time

import httpx
import streamlit as st

import os
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(
    page_title="Sid Sloth",
    page_icon="📈",
    layout="centered",
)

# ── Persistent auth via URL query params ──────────────────────────────────────
# Token is stored in ?t=<jwt> so it survives page refreshes.
# Users can bookmark the URL to stay logged in.

def _load_token_from_url() -> str | None:
    return st.query_params.get("t") or None

def _save_token_to_url(token: str) -> None:
    st.query_params["t"] = token

def _clear_token_from_url() -> None:
    st.query_params.clear()

def _save_page_to_url(page: str) -> None:
    if page == "chat":
        if "p" in st.query_params:
            del st.query_params["p"]
    else:
        st.query_params["p"] = page

def _load_page_from_url() -> str:
    return st.query_params.get("p") or "chat"

def _decode_jwt_is_admin(token: str) -> bool:
    """Decode JWT payload (no validation) to extract is_admin flag."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return bool(payload.get("is_admin", False))
    except Exception:
        return False

# ── Session State Defaults ────────────────────────────────────────────────────

if "token" not in st.session_state:
    st.session_state.token = _load_token_from_url()
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "is_admin" not in st.session_state:
    # Restore from JWT on page refresh — no extra API call needed
    token = st.session_state.token
    st.session_state.is_admin = _decode_jwt_is_admin(token) if token else False
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    # Restore page from URL so admin page survives refresh
    st.session_state.page = _load_page_from_url() if st.session_state.token else "chat"
# Admin ingestion task tracking — restored from backend on page load
if "patreon_task_id" not in st.session_state:
    st.session_state.patreon_task_id = None
if "excel_task_id" not in st.session_state:
    st.session_state.excel_task_id = None
if "doc_task_id" not in st.session_state:
    st.session_state.doc_task_id = None
if "tasks_restored" not in st.session_state:
    st.session_state.tasks_restored = False


# ── API Helpers ───────────────────────────────────────────────────────────────

def api_login(username: str, password: str) -> dict | None:
    try:
        with httpx.Client(trust_env=False, timeout=30.0) as client:
            resp = client.post(
                f"{API_BASE}/auth/login",
                json={"username": username, "password": password},
            )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.json().get("detail", "Login failed")}
    except httpx.ConnectError:
        return {"error": "Cannot connect to server. Is it running?"}


def api_chat(message: str, conversation_id: str | None, token: str) -> dict | None:
    try:
        with httpx.Client(trust_env=False, timeout=90.0) as client:
            resp = client.post(
                f"{API_BASE}/chat",
                json={"message": message, "conversation_id": conversation_id},
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json()
        try:
            detail = resp.json().get("detail", "Unknown error")
        except Exception:
            detail = f"HTTP {resp.status_code}: {resp.text[:200] or 'empty response'}"
        return {"error": detail}
    except httpx.ConnectError:
        return {"error": "Cannot connect to server. Is it running?"}
    except httpx.TimeoutException:
        return {"error": "Request timed out — LLM is thinking."}


def api_start_patreon_ingest(post_url_or_id: str, token: str, force: bool = False) -> dict | None:
    try:
        with httpx.Client(trust_env=False, timeout=15.0) as client:
            resp = client.post(
                f"{API_BASE}/admin/ingest/patreon",
                json={"post_url_or_id": post_url_or_id, "force": force},
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json()
        try:
            detail = resp.json().get("detail", "Unknown error")
        except Exception:
            detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
        return {"error": detail}
    except httpx.ConnectError:
        return {"error": "Cannot connect to server."}


def api_get_task_status(task_id: str, token: str) -> dict | None:
    try:
        with httpx.Client(trust_env=False, timeout=10.0) as client:
            resp = client.get(
                f"{API_BASE}/admin/ingest/status/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def api_get_active_tasks(token: str) -> list[dict]:
    """Fetch all running/recent tasks from backend."""
    try:
        with httpx.Client(trust_env=False, timeout=10.0) as client:
            resp = client.get(
                f"{API_BASE}/admin/ingest/active",
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def api_start_excel_ingest(file_bytes: bytes, filename: str, token: str) -> dict | None:
    """Upload Excel file and start background ingestion. Returns {task_id} or {error}."""
    try:
        with httpx.Client(trust_env=False, timeout=60.0) as client:
            resp = client.post(
                f"{API_BASE}/admin/ingest/excel",
                files={"file": (filename, file_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json()
        try:
            detail = resp.json().get("detail", "Unknown error")
        except Exception:
            detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
        return {"error": detail}
    except httpx.ConnectError:
        return {"error": "Cannot connect to server."}
    except httpx.TimeoutException:
        return {"error": "Server did not respond."}


def api_start_doc_ingest(file_bytes: bytes, filename: str, token: str) -> dict | None:
    """Upload document and start background ingestion. Returns {task_id} or {error}."""
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "application/pdf" if ext == "pdf" else "text/plain"
    try:
        with httpx.Client(trust_env=False, timeout=60.0) as client:
            resp = client.post(
                f"{API_BASE}/admin/ingest/doc",
                files={"file": (filename, file_bytes, mime)},
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json()
        try:
            detail = resp.json().get("detail", "Unknown error")
        except Exception:
            detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
        return {"error": detail}
    except httpx.ConnectError:
        return {"error": "Cannot connect to server."}
    except httpx.TimeoutException:
        return {"error": "Server did not respond."}


# ── Login Page ────────────────────────────────────────────────────────────────

def render_login():
    st.title("Sid Sloth")
    st.caption("Private investment analysis assistant")
    st.divider()

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="your name")
        password = st.text_input("Password", type="password", placeholder="shared password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Enter both username and password.")
            return

        with st.spinner("Authenticating..."):
            result = api_login(username, password)

        if result and "error" not in result:
            st.session_state.token = result["access_token"]
            st.session_state.user_id = result["user_id"]
            st.session_state.is_admin = result.get("is_admin", False)
            _save_token_to_url(result["access_token"])
            st.rerun()
        else:
            err = result.get("error", "Login failed") if result else "Server unreachable"
            st.error(err)


# ── Chat Page ─────────────────────────────────────────────────────────────────

def render_chat():
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.title("Sid Sloth")
        st.caption("Private investment analysis assistant")
    with col2:
        if st.session_state.is_admin:
            if st.button("⚙️ Admin", use_container_width=True):
                st.session_state.page = "admin"
                _save_page_to_url("admin")
                st.rerun()
    with col3:
        if st.button("Sign out", use_container_width=True):
            st.session_state.token = None
            st.session_state.user_id = None
            st.session_state.is_admin = False
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.session_state.page = "chat"
            _clear_token_from_url()
            st.rerun()

    st.divider()

    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "_Ask about a covered stock (e.g. \"What's NVDA looking like?\"), "
                "your portfolio, or the investment methodology._"
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    # Escape $ signs to prevent Streamlit from interpreting
                    # currency values as LaTeX math delimiters.
                    safe_content = msg["content"].replace("$", r"\$")
                    st.markdown(safe_content)
                    if msg["role"] == "assistant" and msg.get("meta"):
                        meta = msg["meta"]
                        cols = st.columns(3)
                        cols[0].caption(f"Model: {meta.get('model_used', '—')}")
                        cols[1].caption(f"Tokens: {meta.get('tokens_used', '—')}")
                        if meta.get("is_degraded"):
                            cols[2].caption("⚡ Economy mode")

    if st.session_state.messages:
        if st.button("New conversation", use_container_width=False):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

    user_input = st.chat_input("Ask about a stock, your portfolio, or the methodology...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "meta": {}})

        with st.spinner("Thinking..."):
            result = api_chat(
                message=user_input,
                conversation_id=st.session_state.conversation_id,
                token=st.session_state.token,
            )

        if result and "error" not in result:
            st.session_state.conversation_id = result["conversation_id"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["reply"],
                "meta": {
                    "model_used": result.get("model_used"),
                    "tokens_used": result.get("tokens_used"),
                    "is_degraded": result.get("is_degraded", False),
                },
            })
        else:
            err = result.get("error", "Unknown error") if result else "No response"
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {err}",
                "meta": {},
            })

        st.rerun()


# ── Admin Page ────────────────────────────────────────────────────────────────

def _task_progress(messages: list[str], key: str) -> None:
    """Collapsed expander showing last 30 progress lines."""
    with st.expander("Progress log", expanded=False):
        for line in messages[-30:]:
            st.text(f"  {line}")


def _restore_active_tasks():
    """On first load, query backend for any running/recent tasks and restore session state."""
    if st.session_state.tasks_restored:
        return
    st.session_state.tasks_restored = True
    token = st.session_state.token
    if not token:
        return
    active = api_get_active_tasks(token)
    for t in active:
        if t["status"] != "running":
            continue
        tid = t["task_id"]
        tt = t.get("task_type", "patreon")
        if tt == "patreon" and not st.session_state.patreon_task_id:
            st.session_state.patreon_task_id = tid
        elif tt == "excel" and not st.session_state.excel_task_id:
            st.session_state.excel_task_id = tid
        elif tt == "doc" and not st.session_state.doc_task_id:
            st.session_state.doc_task_id = tid


def render_admin():
    _restore_active_tasks()

    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.title("⚙️ Admin — Ingestion")
    with col2:
        if st.button("← Chat", use_container_width=True):
            st.session_state.page = "chat"
            _save_page_to_url("chat")
            st.rerun()
    with col3:
        if st.button("Sign out", use_container_width=True):
            st.session_state.token = None
            st.session_state.is_admin = False
            st.session_state.page = "chat"
            _clear_token_from_url()
            st.rerun()

    st.divider()

    # Track whether any section has a running task (rerun fired at very end)
    needs_rerun = False

    # ── Patreon post ingestion ────────────────────────────────────────────────
    st.subheader("Patreon Post")
    st.caption(
        "Paste a Patreon post URL or numeric post ID. "
        "Fetches content, analyzes charts, extracts trade signals, and stores chunks."
    )

    task_id = st.session_state.patreon_task_id

    if task_id:
        status = api_get_task_status(task_id, st.session_state.token)
        if status is None:
            st.warning("Task not found — server may have restarted.")
            if st.button("Dismiss", key="dismiss_lost"):
                st.session_state.patreon_task_id = None

                st.rerun()
        elif status["status"] == "running":
            st.info("Ingestion in progress…")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "patreon_prog")
            else:
                st.caption("Starting…")
            if st.button("Cancel", key="patreon_cancel"):
                st.session_state.patreon_task_id = None

                st.rerun()
            needs_rerun = True
        elif status["status"] == "done":
            result = status.get("result", {})
            st.success("Ingestion complete!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks", result.get("chunk_count", 0))
            c2.metric("Signals", result.get("signal_count", 0))
            c3.metric("Charts", result.get("image_count", 0))
            st.caption(f"Post: **{result.get('title', '')}**")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "patreon_done_prog")
            if st.button("Ingest another post", key="patreon_reset"):
                st.session_state.patreon_task_id = None

                st.rerun()
        elif status["status"] == "error":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "patreon_err_prog")
            if st.button("Try again", key="patreon_retry"):
                st.session_state.patreon_task_id = None

                st.rerun()
    else:
        with st.form("patreon_ingest_form"):
            post_input = st.text_input(
                "Post URL or ID",
                placeholder="https://www.patreon.com/posts/oil-war-154313150  or  154313150",
            )
            force = st.checkbox(
                "Force re-ingest", value=False,
                help="Wipe existing chunks/signals and re-process from scratch.",
            )
            submitted = st.form_submit_button("Ingest Post", use_container_width=True)

        if submitted:
            if not post_input.strip():
                st.error("Enter a post URL or ID.")
            else:
                resp = api_start_patreon_ingest(post_input.strip(), st.session_state.token, force=force)
                if not resp or "error" in resp:
                    st.error(resp.get("error", "Failed to start ingestion") if resp else "No response")
                else:
                    st.session_state.patreon_task_id = resp["task_id"]
                    pass  # task_id saved to session_state above
                    st.rerun()

    st.divider()

    # ── Excel workbook ingestion ──────────────────────────────────────────────
    st.subheader("Excel Workbook (.xlsx)")
    st.caption(
        "Upload the analyst's Excel workbook. Rephrases and upserts stock predictions "
        "and principles — marks previous predictions as superseded."
    )

    excel_task_id = st.session_state.excel_task_id

    if excel_task_id:
        status = api_get_task_status(excel_task_id, st.session_state.token)
        if status is None:
            st.warning("Task not found — server may have restarted.")
            st.session_state.excel_task_id = None
        elif status["status"] == "running":
            st.info("Excel ingestion in progress…")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "excel_prog")
            else:
                st.caption("Starting…")
            if st.button("Cancel", key="excel_cancel"):
                st.session_state.excel_task_id = None
                st.rerun()
            needs_rerun = True
        elif status["status"] == "done":
            result = status.get("result", {})
            st.success("Excel ingestion complete!")
            c1, c2 = st.columns(2)
            c1.metric("Stocks upserted", result.get("stock_count", 0))
            c2.metric("Principles", result.get("principle_count", 0))
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "excel_done_prog")
            if st.button("Upload another workbook", key="excel_reset"):
                st.session_state.excel_task_id = None
                st.rerun()
        elif status["status"] == "error":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "excel_err_prog")
            if st.button("Try again", key="excel_retry"):
                st.session_state.excel_task_id = None
                st.rerun()
    else:
        excel_file = st.file_uploader("Choose .xlsx file", type=["xlsx"], key="excel_uploader")
        if excel_file is not None:
            if st.button("Ingest Excel", key="excel_btn"):
                with st.spinner("Uploading…"):
                    resp = api_start_excel_ingest(excel_file.read(), excel_file.name, st.session_state.token)
                if resp and "task_id" in resp:
                    st.session_state.excel_task_id = resp["task_id"]
                    st.rerun()
                else:
                    st.error(resp.get("error", "Unknown error") if resp else "No response")

    st.divider()

    # ── PDF / text document ingestion ─────────────────────────────────────────
    st.subheader("Document (PDF / Text)")
    st.caption(
        "Upload a PDF or text file. Chunks, rephrases, embeds, and stores as analyst knowledge."
    )

    doc_task_id = st.session_state.doc_task_id

    if doc_task_id:
        status = api_get_task_status(doc_task_id, st.session_state.token)
        if status is None:
            st.warning("Task not found — server may have restarted.")
            st.session_state.doc_task_id = None
        elif status["status"] == "running":
            st.info("Document ingestion in progress…")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "doc_prog")
            else:
                st.caption("Starting…")
            if st.button("Cancel", key="doc_cancel"):
                st.session_state.doc_task_id = None
                st.rerun()
            needs_rerun = True
        elif status["status"] == "done":
            result = status.get("result", {})
            st.success("Document ingestion complete!")
            st.metric("Chunks stored", result.get("chunk_count", 0))
            st.caption(f"File: **{result.get('file_name', '')}**")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "doc_done_prog")
            if st.button("Upload another document", key="doc_reset"):
                st.session_state.doc_task_id = None
                st.rerun()
        elif status["status"] == "error":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages, "doc_err_prog")
            if st.button("Try again", key="doc_retry"):
                st.session_state.doc_task_id = None
                st.rerun()
    else:
        doc_file = st.file_uploader("Choose PDF or text file", type=["pdf", "txt", "md"], key="doc_uploader")
        if doc_file is not None:
            if st.button("Ingest Document", key="doc_btn"):
                with st.spinner("Uploading…"):
                    resp = api_start_doc_ingest(doc_file.read(), doc_file.name, st.session_state.token)
                if resp and "task_id" in resp:
                    st.session_state.doc_task_id = resp["task_id"]
                    st.rerun()
                else:
                    st.error(resp.get("error", "Unknown error") if resp else "No response")

    st.divider()

    # ── Session cookie refresh reminder ──────────────────────────────────────
    st.subheader("Patreon Session Cookie")
    st.caption(
        "The `PATREON_SESSION_ID` expires every ~2 weeks. "
        "Refresh: log into Patreon → DevTools → Application → Cookies → copy `session_id` → update Railway env var."
    )
    st.code("PATREON_SESSION_ID=your_new_session_id_here", language="bash")

    # Auto-refresh fired AFTER all sections are rendered
    if needs_rerun:
        time.sleep(2)
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

if not st.session_state.token:
    # Not logged in — always show login regardless of URL params
    render_login()
elif st.session_state.page == "admin":
    if st.session_state.is_admin:
        render_admin()
    else:
        # Non-admin somehow landed on admin — redirect to chat
        st.session_state.page = "chat"
        _save_page_to_url("chat")
        st.rerun()
else:
    render_chat()
