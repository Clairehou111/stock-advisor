"""
Sid Sloth — Streamlit frontend.

Pages:
  - Login (shared password → JWT)
  - Chat (main chatbot interface)
  - Admin (admin only: trigger ingestion pipelines)
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import quote, unquote

import httpx
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api")
AUTH_COOKIE_NAME = "sid_sloth_auth"
CONVERSATION_COOKIE_NAME = "sid_sloth_conv"
MAX_RECENT_CONVERSATIONS = 12

st.set_page_config(
    page_title="Sid Sloth",
    page_icon="📈",
    layout="centered",
)

CHAT_INPUT_SLOT = st.empty()


def _save_page_to_url(page: str) -> None:
    if page == "chat":
        if "p" in st.query_params:
            del st.query_params["p"]
    else:
        st.query_params["p"] = page


def _load_page_from_url() -> str:
    return st.query_params.get("p") or "chat"


def _next_js_key(prefix: str) -> str:
    nonce = st.session_state.get("_js_nonce", 0) + 1
    st.session_state._js_nonce = nonce
    return f"{prefix}_{nonce}"


def _apply_verified_auth(token: str, user_id: str, is_admin: bool) -> None:
    st.session_state.token = token
    st.session_state.user_id = user_id
    st.session_state.is_admin = is_admin
    st.session_state._signed_out = False


def _ensure_state_defaults() -> None:
    defaults = {
        "token": None,
        "user_id": None,
        "is_admin": False,
        "conversation_id": None,
        "loaded_conversation_id": None,
        "messages": [],
        "page": "chat",
        "recent_conversations": [],
        "recent_conversations_loaded": False,
        "recent_conversations_dirty": False,
        "patreon_task_id": None,
        "excel_task_id": None,
        "doc_task_id": None,
        "tasks_restored": False,
        "active_conversation_notice": None,
        "_pending_cookie_ops": {},
        "_signed_out": False,
        "_auth_cookie_checked": False,
        "_js_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _queue_cookie_set(cookie_name: str, value: str, *, max_age: int = 7 * 24 * 3600) -> None:
    pending = dict(st.session_state.get("_pending_cookie_ops", {}))
    pending[cookie_name] = {"value": value, "max_age": max_age}
    st.session_state._pending_cookie_ops = pending


def _queue_cookie_remove(*cookie_names: str) -> None:
    pending = dict(st.session_state.get("_pending_cookie_ops", {}))
    for cookie_name in cookie_names:
        pending[cookie_name] = None
    st.session_state._pending_cookie_ops = pending


def _flush_cookie_ops() -> None:
    pending = dict(st.session_state.get("_pending_cookie_ops", {}))
    if not pending:
        return

    for cookie_name, payload in pending.items():
        cookie_name_js = json.dumps(cookie_name)
        if payload is None:
            js = (
                f"document.cookie = {cookie_name_js} + '=; path=/; Max-Age=0; SameSite=Lax';"
            )
        else:
            cookie_value = json.dumps(quote(payload["value"], safe=""))
            max_age = int(payload["max_age"])
            js = (
                "(() => {"
                f"const secure = window.location.protocol === 'https:' ? '; Secure' : '';"
                f"document.cookie = {cookie_name_js} + '=' + {cookie_value} + '; path=/; Max-Age={max_age}; SameSite=Lax' + secure;"
                "})()"
            )
        streamlit_js_eval(js_expressions=js, key=_next_js_key(f"cookie_{cookie_name}"))

    st.session_state._pending_cookie_ops = {}


def _get_cookie(cookie_name: str) -> str | None:
    try:
        value = st.context.cookies.get(cookie_name)
    except Exception:
        return None
    if not value:
        return None
    try:
        return unquote(value)
    except Exception:
        return value


def _bootstrap_auth_from_storage() -> None:
    if st.session_state.token or st.session_state.get("_signed_out"):
        st.session_state._auth_cookie_checked = True
        return
    cookie_token = _get_cookie(AUTH_COOKIE_NAME)
    if not cookie_token:
        st.session_state._auth_cookie_checked = True
        return
    result = api_auth_me(cookie_token)
    if result and "error" not in result:
        _apply_verified_auth(cookie_token, result["user_id"], result.get("is_admin", False))
        st.session_state.page = _load_page_from_url()
    else:
        _queue_cookie_remove(AUTH_COOKIE_NAME, CONVERSATION_COOKIE_NAME)
        st.session_state._signed_out = True
    st.session_state._auth_cookie_checked = True


def _bootstrap_conversation_from_storage() -> None:
    if not st.session_state.token or st.session_state.conversation_id:
        return
    stored_conv = _get_cookie(CONVERSATION_COOKIE_NAME)
    if stored_conv:
        st.session_state.conversation_id = stored_conv
        st.session_state.loaded_conversation_id = None


def _clear_active_conversation() -> None:
    st.session_state.conversation_id = None
    st.session_state.loaded_conversation_id = None
    st.session_state.messages = []
    st.session_state.active_conversation_notice = None
    _queue_cookie_remove(CONVERSATION_COOKIE_NAME)


def _begin_new_conversation() -> None:
    _clear_active_conversation()


def _activate_conversation(conversation_id: str) -> None:
    st.session_state.conversation_id = conversation_id
    st.session_state.loaded_conversation_id = None
    st.session_state.messages = []
    st.session_state.active_conversation_notice = None
    _queue_cookie_set(CONVERSATION_COOKIE_NAME, conversation_id)


def _sign_out() -> None:
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.is_admin = False
    st.session_state.page = "chat"
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.loaded_conversation_id = None
    st.session_state.recent_conversations = []
    st.session_state.recent_conversations_loaded = False
    st.session_state.recent_conversations_dirty = False
    st.session_state.tasks_restored = False
    st.session_state.patreon_task_id = None
    st.session_state.excel_task_id = None
    st.session_state.doc_task_id = None
    st.session_state._signed_out = True
    st.session_state._auth_cookie_checked = True
    st.session_state.active_conversation_notice = None
    _queue_cookie_remove(AUTH_COOKIE_NAME, CONVERSATION_COOKIE_NAME)
    st.query_params.clear()


def _extract_error(resp: httpx.Response) -> str:
    try:
        return resp.json().get("detail", "Unknown error")
    except Exception:
        return f"HTTP {resp.status_code}: {resp.text[:300] or 'empty response'}"


def _api_request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    timeout: float = 30.0,
    **kwargs,
) -> dict | list | None:
    headers = dict(kwargs.pop("headers", {}))
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            resp = client.request(
                method,
                f"{API_BASE}{path}",
                headers=headers,
                **kwargs,
            )
    except httpx.ConnectError:
        return {"error": "Cannot connect to server. Is it running?", "status_code": 0}
    except httpx.TimeoutException:
        return {"error": "Server did not respond in time.", "status_code": 0}

    if resp.status_code == 200:
        return resp.json()

    return {"error": _extract_error(resp), "status_code": resp.status_code}


def api_login(username: str, password: str) -> dict | None:
    return _api_request(
        "POST",
        "/auth/login",
        json={"username": username, "password": password},
    )


def api_auth_me(token: str) -> dict | None:
    return _api_request(
        "GET",
        "/auth/me",
        token=token,
        timeout=15.0,
    )


def api_chat(message: str, conversation_id: str | None, token: str) -> dict | None:
    result = _api_request(
        "POST",
        "/chat",
        token=token,
        timeout=90.0,
        json={"message": message, "conversation_id": conversation_id},
    )
    if result and isinstance(result, dict) and result.get("status_code") == 0:
        result["error"] = "Request timed out or server unreachable."
    return result


def api_get_conversations(token: str) -> dict | None:
    return _api_request(
        "GET",
        f"/conversations?limit={MAX_RECENT_CONVERSATIONS}",
        token=token,
        timeout=15.0,
    )


def api_get_conversation(conversation_id: str, token: str) -> dict | None:
    return _api_request(
        "GET",
        f"/conversations/{conversation_id}?limit=100",
        token=token,
        timeout=15.0,
    )


def api_start_patreon_ingest(post_url_or_id: str, token: str, force: bool = False) -> dict | None:
    return _api_request(
        "POST",
        "/admin/ingest/patreon",
        token=token,
        timeout=15.0,
        json={"post_url_or_id": post_url_or_id, "force": force},
    )


def api_get_task_status(task_id: str, token: str) -> dict | None:
    return _api_request(
        "GET",
        f"/admin/ingest/status/{task_id}",
        token=token,
        timeout=10.0,
    )


def api_get_active_tasks(token: str) -> list[dict]:
    result = _api_request(
        "GET",
        "/admin/ingest/active",
        token=token,
        timeout=10.0,
    )
    _handle_auth_error(result if isinstance(result, dict) else None)
    if isinstance(result, dict) and "error" in result:
        return []
    return result or []


def api_start_excel_ingest(file_bytes: bytes, filename: str, token: str) -> dict | None:
    return _api_request(
        "POST",
        "/admin/ingest/excel",
        token=token,
        timeout=60.0,
        files={
            "file": (
                filename,
                file_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def api_start_doc_ingest(file_bytes: bytes, filename: str, token: str) -> dict | None:
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "application/pdf" if ext == "pdf" else "text/plain"
    return _api_request(
        "POST",
        "/admin/ingest/doc",
        token=token,
        timeout=60.0,
        files={"file": (filename, file_bytes, mime)},
    )


def _handle_auth_error(result: dict | None) -> bool:
    if isinstance(result, dict) and result.get("status_code") == 401:
        _sign_out()
        st.rerun()
    return False


def _format_message_record(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "role": record.get("role", "assistant"),
        "content": record.get("content", ""),
        "meta": {
            "model_used": record.get("model_used"),
            "tokens_used": record.get("tokens_used"),
        },
    }


def _refresh_recent_conversations(force: bool = False) -> None:
    if not st.session_state.token:
        return
    if (
        not force
        and st.session_state.recent_conversations_loaded
        and not st.session_state.recent_conversations_dirty
    ):
        return

    result = api_get_conversations(st.session_state.token)
    _handle_auth_error(result)
    if isinstance(result, dict) and "error" in result:
        return

    st.session_state.recent_conversations = (result or {}).get("conversations", [])
    st.session_state.recent_conversations_loaded = True
    st.session_state.recent_conversations_dirty = False


def _hydrate_active_conversation(force: bool = False) -> None:
    conv_id = st.session_state.conversation_id
    if not st.session_state.token or not conv_id:
        return
    if not force and st.session_state.loaded_conversation_id == conv_id:
        return

    result = api_get_conversation(conv_id, st.session_state.token)
    _handle_auth_error(result)
    if isinstance(result, dict) and result.get("status_code") == 404:
        _begin_new_conversation()
        st.session_state.recent_conversations_dirty = True
        return
    if isinstance(result, dict) and "error" in result:
        return

    st.session_state.messages = [
        _format_message_record(message)
        for message in (result or {}).get("messages", [])
    ]
    total_count = (result or {}).get("total_message_count", len(st.session_state.messages))
    if (result or {}).get("truncated"):
        st.session_state.active_conversation_notice = (
            f"Showing the most recent {len(st.session_state.messages)} of {total_count} messages."
        )
    else:
        st.session_state.active_conversation_notice = None
    st.session_state.loaded_conversation_id = conv_id


def _format_conversation_stamp(raw_value: str | None) -> str:
    if not raw_value:
        return ""
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).strftime("%b %d %H:%M")
    except Exception:
        return ""


def _short_title(title: str | None) -> str:
    title = (title or "New conversation").strip()
    if len(title) <= 34:
        return title
    return title[:31] + "..."


def _render_conversation_sidebar() -> None:
    _refresh_recent_conversations()

    with st.sidebar:
        st.subheader("Conversations")
        if st.button("New conversation", use_container_width=True):
            _begin_new_conversation()
            st.rerun()

        conversations = st.session_state.recent_conversations
        if not conversations:
            st.caption("No saved conversations yet.")
            return

        for conv in conversations:
            conv_id = conv.get("id")
            title = _short_title(conv.get("title"))
            button_type = "primary" if conv_id == st.session_state.conversation_id else "secondary"
            if st.button(
                title,
                key=f"open_conv_{conv_id}",
                use_container_width=True,
                type=button_type,
            ):
                _activate_conversation(conv_id)
                st.rerun()

            details = []
            stamp = _format_conversation_stamp(
                conv.get("last_message_at") or conv.get("created_at")
            )
            if stamp:
                details.append(stamp)
            message_count = conv.get("message_count")
            if message_count:
                details.append(f"{message_count} msgs")
            if details:
                st.caption(" | ".join(details))


def render_login() -> None:
    CHAT_INPUT_SLOT.empty()
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
            _apply_verified_auth(
                result["access_token"],
                result["user_id"],
                result.get("is_admin", False),
            )
            st.session_state.page = "chat"
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.loaded_conversation_id = None
            st.session_state.recent_conversations = []
            st.session_state.recent_conversations_loaded = False
            st.session_state.recent_conversations_dirty = True
            st.session_state.active_conversation_notice = None
            _queue_cookie_set(AUTH_COOKIE_NAME, result["access_token"])
            _queue_cookie_remove(CONVERSATION_COOKIE_NAME)
            st.rerun()

        err = result.get("error", "Login failed") if result else "Server unreachable"
        st.error(err)


def render_restoring_session() -> None:
    CHAT_INPUT_SLOT.empty()
    st.title("Sid Sloth")
    st.caption("Restoring your session...")


def render_chat() -> None:
    _hydrate_active_conversation()
    _render_conversation_sidebar()

    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.title("Sid Sloth")
        st.caption("Private investment analysis assistant")
    with col2:
        if st.session_state.is_admin and st.button("⚙️ Admin", use_container_width=True):
            st.session_state.page = "admin"
            _save_page_to_url("admin")
            st.rerun()
    with col3:
        if st.button("Sign out", use_container_width=True):
            _sign_out()
            st.rerun()

    st.divider()

    if st.session_state.active_conversation_notice:
        st.caption(st.session_state.active_conversation_notice)

    if not st.session_state.messages:
        st.caption("Start with a stock, portfolio, or methodology question.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            safe_content = msg["content"].replace("$", r"\$")
            st.markdown(safe_content)
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                cols = st.columns(3)
                cols[0].caption(f"Model: {meta.get('model_used', '—')}")
                cols[1].caption(f"Tokens: {meta.get('tokens_used', '—')}")
                if meta.get("is_degraded"):
                    cols[2].caption("⚡ Economy mode")

    user_input = CHAT_INPUT_SLOT.chat_input(
        "Ask about a stock, your portfolio, or the methodology..."
    )
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input, "meta": {}})

    with st.spinner("Thinking..."):
        result = api_chat(
            message=user_input,
            conversation_id=st.session_state.conversation_id,
            token=st.session_state.token,
        )

    _handle_auth_error(result)
    if result and "error" not in result:
        conversation_id = result["conversation_id"]
        st.session_state.conversation_id = conversation_id
        st.session_state.loaded_conversation_id = conversation_id
        st.session_state.active_conversation_notice = None
        _queue_cookie_set(CONVERSATION_COOKIE_NAME, conversation_id)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["reply"],
                "meta": {
                    "model_used": result.get("model_used"),
                    "tokens_used": result.get("tokens_used"),
                    "is_degraded": result.get("is_degraded", False),
                },
            }
        )
        st.session_state.recent_conversations_dirty = True
    else:
        err = result.get("error", "Unknown error") if result else "No response"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"⚠️ {err}",
                "meta": {},
            }
        )

    st.rerun()


def _task_progress(messages: list[str]) -> None:
    with st.expander("Progress log", expanded=False):
        for line in messages[-30:]:
            st.text(f"  {line}")


def _restore_active_tasks() -> None:
    if st.session_state.tasks_restored:
        return
    st.session_state.tasks_restored = True
    token = st.session_state.token
    if not token:
        return
    active = api_get_active_tasks(token)
    for task in active:
        if task.get("status") != "running":
            continue
        task_id = task["task_id"]
        task_type = task.get("task_type", "patreon")
        if task_type == "patreon" and not st.session_state.patreon_task_id:
            st.session_state.patreon_task_id = task_id
        elif task_type == "excel" and not st.session_state.excel_task_id:
            st.session_state.excel_task_id = task_id
        elif task_type == "doc" and not st.session_state.doc_task_id:
            st.session_state.doc_task_id = task_id


def render_admin() -> None:
    CHAT_INPUT_SLOT.empty()
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
            _sign_out()
            st.rerun()

    st.divider()

    needs_rerun = False

    st.subheader("Patreon Post")
    st.caption(
        "Paste a Patreon post URL or numeric post ID. "
        "Fetches content, analyzes charts, extracts trade signals, and stores chunks."
    )

    task_id = st.session_state.patreon_task_id
    if task_id:
        status = api_get_task_status(task_id, st.session_state.token)
        _handle_auth_error(status if isinstance(status, dict) else None)
        if not status:
            st.error("Could not load task status.")
        elif status.get("status_code") == 404:
            st.warning("Task not found — server may have restarted.")
            if st.button("Dismiss", key="dismiss_lost"):
                st.session_state.patreon_task_id = None
                st.rerun()
        elif status.get("error"):
            st.error(status["error"])
        elif status["status"] == "running":
            st.info("Ingestion in progress…")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages)
            else:
                st.caption("Starting…")
            if st.button("Cancel", key="patreon_cancel"):
                st.session_state.patreon_task_id = None
                st.rerun()
            needs_rerun = True
        elif status["status"] == "done":
            result = status.get("result", {})
            post_title = result.get("title") or "Untitled Patreon post"
            st.success("Ingestion complete!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks", result.get("chunk_count", 0))
            c2.metric("Signals", result.get("signal_count", 0))
            c3.metric("Charts", result.get("image_count", 0))
            st.caption(f"Post: **{post_title}**")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages)
            if st.button("Ingest another post", key="patreon_reset"):
                st.session_state.patreon_task_id = None
                st.rerun()
        elif status["status"] == "error":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages)
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
                "Force re-ingest",
                value=False,
                help="Wipe existing chunks/signals and re-process from scratch.",
            )
            submitted = st.form_submit_button("Ingest Post", use_container_width=True)

        if submitted:
            if not post_input.strip():
                st.error("Enter a post URL or ID.")
            else:
                resp = api_start_patreon_ingest(
                    post_input.strip(),
                    st.session_state.token,
                    force=force,
                )
                _handle_auth_error(resp)
                if not resp or "error" in resp:
                    st.error(resp.get("error", "Failed to start ingestion") if resp else "No response")
                else:
                    st.session_state.patreon_task_id = resp["task_id"]
                    st.rerun()

    st.divider()

    st.subheader("Excel Workbook (.xlsx)")
    st.caption(
        "Upload the analyst's Excel workbook. Rephrases and upserts stock predictions "
        "and principles — marks previous predictions as superseded."
    )

    excel_task_id = st.session_state.excel_task_id
    if excel_task_id:
        status = api_get_task_status(excel_task_id, st.session_state.token)
        _handle_auth_error(status if isinstance(status, dict) else None)
        if not status:
            st.error("Could not load task status.")
        elif status.get("status_code") == 404:
            st.warning("Task not found — server may have restarted.")
            st.session_state.excel_task_id = None
        elif status.get("error"):
            st.error(status["error"])
        elif status["status"] == "running":
            st.info("Excel ingestion in progress…")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages)
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
                _task_progress(messages)
            if st.button("Upload another workbook", key="excel_reset"):
                st.session_state.excel_task_id = None
                st.rerun()
        elif status["status"] == "error":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages)
            if st.button("Try again", key="excel_retry"):
                st.session_state.excel_task_id = None
                st.rerun()
    else:
        excel_file = st.file_uploader("Choose .xlsx file", type=["xlsx"], key="excel_uploader")
        if excel_file is not None and st.button("Ingest Excel", key="excel_btn"):
            with st.spinner("Uploading…"):
                resp = api_start_excel_ingest(excel_file.read(), excel_file.name, st.session_state.token)
            _handle_auth_error(resp)
            if resp and "task_id" in resp:
                st.session_state.excel_task_id = resp["task_id"]
                st.rerun()
            else:
                st.error(resp.get("error", "Unknown error") if resp else "No response")

    st.divider()

    st.subheader("Document (PDF / Text)")
    st.caption("Upload a PDF or text file. Chunks, rephrases, embeds, and stores as analyst knowledge.")

    doc_task_id = st.session_state.doc_task_id
    if doc_task_id:
        status = api_get_task_status(doc_task_id, st.session_state.token)
        _handle_auth_error(status if isinstance(status, dict) else None)
        if not status:
            st.error("Could not load task status.")
        elif status.get("status_code") == 404:
            st.warning("Task not found — server may have restarted.")
            st.session_state.doc_task_id = None
        elif status.get("error"):
            st.error(status["error"])
        elif status["status"] == "running":
            st.info("Document ingestion in progress…")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages)
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
                _task_progress(messages)
            if st.button("Upload another document", key="doc_reset"):
                st.session_state.doc_task_id = None
                st.rerun()
        elif status["status"] == "error":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")
            messages = status.get("messages", [])
            if messages:
                _task_progress(messages)
            if st.button("Try again", key="doc_retry"):
                st.session_state.doc_task_id = None
                st.rerun()
    else:
        doc_file = st.file_uploader("Choose PDF or text file", type=["pdf", "txt", "md"], key="doc_uploader")
        if doc_file is not None and st.button("Ingest Document", key="doc_btn"):
            with st.spinner("Uploading…"):
                resp = api_start_doc_ingest(doc_file.read(), doc_file.name, st.session_state.token)
            _handle_auth_error(resp)
            if resp and "task_id" in resp:
                st.session_state.doc_task_id = resp["task_id"]
                st.rerun()
            else:
                st.error(resp.get("error", "Unknown error") if resp else "No response")

    st.divider()

    st.subheader("Patreon Session Cookie")
    st.caption(
        "The `PATREON_SESSION_ID` expires every ~2 weeks. "
        "Refresh: log into Patreon → DevTools → Application → Cookies → copy `session_id` → update Railway env var."
    )
    st.code("PATREON_SESSION_ID=your_new_session_id_here", language="bash")

    if needs_rerun:
        time.sleep(2)
        st.rerun()


_ensure_state_defaults()
_flush_cookie_ops()
_bootstrap_auth_from_storage()
_bootstrap_conversation_from_storage()
_flush_cookie_ops()

if not st.session_state.token:
    if not st.session_state.get("_signed_out") and not st.session_state._auth_cookie_checked:
        render_restoring_session()
    else:
        render_login()
elif st.session_state.page == "admin":
    if st.session_state.is_admin:
        render_admin()
    else:
        st.session_state.page = "chat"
        _save_page_to_url("chat")
        st.rerun()
else:
    render_chat()
