"""Keep ``__Secure-1PSIDTS`` fresh so a stored session does not expire in ~an hour.

Google rotates ``__Secure-1PSIDTS`` and stops honouring the previous value. A
browser picks the new one up because it keeps calling
``accounts.google.com/RotateCookies``; a file-backed cookie jar has to do the
same or the session silently degrades to logged-out. That degradation is easy to
miss: anonymous Gemini still answers text prompts, so only the paths that need a
real account (image input, for one) start failing.

Note that Gemini's own responses do *not* carry a rotated ``__Secure-1PSIDTS`` --
only ``NID`` -- so passively watching ``Set-Cookie`` on StreamGenerate replies is
not enough. The rotation endpoint has to be called explicitly.
"""
from __future__ import annotations

import errno
import json
import os
import stat
import threading
import urllib.request

from .config import CONFIG
from .gemini import (
    HAS_CURL_CFFI,
    _cookie_cache,
    _get_ssl_ctx,
    curl_requests,
    load_cookie,
    log,
)

ROTATE_COOKIES_URL = "https://accounts.google.com/RotateCookies"
ROTATE_COOKIES_BODY = json.dumps([0, "-0000000000000000000"])
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Only session-rotating cookies are written back; everything else in the jar is
# long-lived and rewriting it would risk clobbering a good value with a stale one.
ROTATING_COOKIE_NAMES = ("__Secure-1PSIDTS", "__Secure-3PSIDTS")

DEFAULT_REFRESH_INTERVAL_SEC = 540
# The endpoint answers a dead session with 401 and no new cookie; slow down
# instead of hammering it every interval.
FAILURE_BACKOFF_MULTIPLIER = 4

_write_lock = threading.Lock()
_refresher_thread: threading.Thread | None = None


def parse_set_cookie_pairs(raw_values) -> dict:
    """Pull ``name -> value`` out of raw Set-Cookie header lines."""
    pairs = {}
    for raw in raw_values:
        if not raw:
            continue
        first = raw.split(";", 1)[0].strip()
        if "=" not in first:
            continue
        name, value = first.split("=", 1)
        name, value = name.strip(), value.strip()
        # An expiry-style deletion carries an empty value; keep the real one.
        if name and value:
            pairs[name] = value
    return pairs


def merge_cookie_string(cookie_str: str, updates: dict) -> str:
    """Replace updated cookies in place and append genuinely new ones."""
    if not updates:
        return cookie_str
    parts = [part.strip() for part in cookie_str.split(";") if part.strip()]
    seen = set()
    merged = []
    for part in parts:
        if "=" not in part:
            merged.append(part)
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name in updates:
            merged.append(f"{name}={updates[name]}")
            seen.add(name)
        else:
            merged.append(f"{name}={value.strip()}")
    for name, value in updates.items():
        if name not in seen:
            merged.append(f"{name}={value}")
    return "; ".join(merged)


_OWNER_ONLY = stat.S_IRUSR | stat.S_IWUSR
# os.replace() cannot swap a bind-mounted file: the mount point itself is busy.
# That is the normal Docker layout (-v ./gemini-auth.json:/app/gemini-auth.json),
# so the rewrite has to fall back to the existing inode rather than give up.
_REPLACE_FALLBACK_ERRNOS = (errno.EBUSY, errno.EXDEV, errno.EACCES, errno.EPERM)


def _write_cookie_file(path: str, payload: str) -> None:
    """Replace atomically where possible, else rewrite the file in place.

    The temp file holds a full copy of the session, so it is removed on every
    path -- including the failure paths -- rather than left in the directory.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp")
    try:
        with open(tmp_path, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, _OWNER_ONLY)
        os.replace(tmp_path, path)
        return
    except OSError as exc:
        if exc.errno not in _REPLACE_FALLBACK_ERRNOS:
            raise
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    with open(path, "r+") as handle:
        handle.seek(0)
        handle.write(payload)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(path, _OWNER_ONLY)
    except OSError:
        pass


def persist_rotated_cookies(updates: dict) -> str:
    """Merge ``updates`` into the configured cookie file.

    Returns a status: ``written``, ``unchanged``, ``no_file``, ``read_failed``
    or ``write_failed``. A failed write is reported distinctly from an unchanged
    one so a read-only cookie file cannot masquerade as "nothing to do".

    Re-reads the file inside the lock so a concurrent update from the browser
    extension is merged into rather than overwritten.
    """
    if not updates:
        return "unchanged"
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return "no_file"

    with _write_lock:
        try:
            with open(cookie_file, "r") as handle:
                content = handle.read().strip()
        except OSError as exc:
            log(f"Cookie refresh: cannot read {cookie_file}: {exc}")
            return "read_failed"

        is_json = content.startswith("{")
        if is_json:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                log(f"Cookie refresh: cannot parse {cookie_file}: {exc}")
                return "read_failed"
            cookie_str = data.get("cookie", "")
        else:
            data = {}
            cookie_str = content

        merged = merge_cookie_string(cookie_str, updates)
        if merged == cookie_str:
            return "unchanged"

        if is_json:
            data["cookie"] = merged
            payload = json.dumps(data)
        else:
            payload = merged

        try:
            _write_cookie_file(cookie_file, payload)
        except OSError as exc:
            hint = ""
            if exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
                hint = " (mounted read-only? the refresher needs write access)"
            log(f"Cookie refresh: cannot write {cookie_file}: {exc}{hint}")
            return "write_failed"

        # load_cookie() caches on mtime; refresh it so in-flight requests pick
        # the new value up without waiting for the next stat.
        try:
            _cookie_cache.update({
                "str": merged,
                "sapisid": data.get("sapisid") or _cookie_cache.get("sapisid"),
                "mtime": os.path.getmtime(cookie_file),
            })
        except OSError:
            _cookie_cache["mtime"] = 0
        return "written"


def _set_cookie_values(response) -> list:
    """Collect Set-Cookie lines from a curl_cffi or urllib response."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return []
    multi_items = getattr(headers, "multi_items", None)
    if callable(multi_items):
        return [value for key, value in multi_items() if key.lower() == "set-cookie"]
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        return get_all("Set-Cookie") or []
    items = getattr(headers, "items", None)
    if callable(items):
        return [value for key, value in items() if key.lower() == "set-cookie"]
    return []


def rotate_cookies_once() -> tuple:
    """Ask Google for a fresh ``__Secure-1PSIDTS``.

    Returns ``(ok, detail)`` where ``detail`` names what changed or why it did not.
    """
    cookie_str, _ = load_cookie()
    if not cookie_str:
        return False, "no_cookie"

    headers = {
        "Content-Type": "application/json",
        "Cookie": cookie_str,
        "User-Agent": USER_AGENT,
    }
    proxy = CONFIG.get("proxy")

    try:
        if HAS_CURL_CFFI:
            kwargs = {"headers": headers, "timeout": 30, "impersonate": "chrome"}
            if proxy:
                kwargs["proxy"] = proxy
            response = curl_requests.post(ROTATE_COOKIES_URL, data=ROTATE_COOKIES_BODY, **kwargs)
            status = response.status_code
            set_cookies = _set_cookie_values(response)
        else:
            request = urllib.request.Request(
                ROTATE_COOKIES_URL, data=ROTATE_COOKIES_BODY.encode(), headers=headers, method="POST"
            )
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
                )
                response = opener.open(request, timeout=30)
            else:
                response = urllib.request.urlopen(request, context=_get_ssl_ctx(), timeout=30)
            status = response.status
            set_cookies = _set_cookie_values(response)
    except Exception as exc:  # network, TLS, or an HTTP error class
        status = getattr(exc, "code", None)
        if status is None:
            return False, f"error:{type(exc).__name__}"
        set_cookies = _set_cookie_values(exc)

    if status == 401:
        # Signed out, or the stored cookies are already too old to rotate.
        return False, "unauthorized"
    if status != 200:
        return False, f"http_{status}"

    pairs = parse_set_cookie_pairs(set_cookies)
    updates = {name: pairs[name] for name in ROTATING_COOKIE_NAMES if name in pairs}
    if not updates:
        return False, "no_rotating_cookie"
    status = persist_rotated_cookies(updates)
    if status != "written":
        return False, status
    return True, ",".join(sorted(updates))


def refresh_interval_sec() -> int:
    try:
        value = int(CONFIG.get("cookie_refresh_interval_sec") or DEFAULT_REFRESH_INTERVAL_SEC)
    except (TypeError, ValueError):
        return DEFAULT_REFRESH_INTERVAL_SEC
    return max(60, value)


def _refresh_loop(stop_event: threading.Event) -> None:
    interval = refresh_interval_sec()
    while not stop_event.is_set():
        ok, detail = rotate_cookies_once()
        if ok:
            log(f"Cookie refresh: rotated {detail}")
            delay = interval
        else:
            if detail not in ("no_cookie", "unchanged"):
                log(f"Cookie refresh: skipped ({detail})")
            delay = interval * FAILURE_BACKOFF_MULTIPLIER
        stop_event.wait(delay)


def start_cookie_refresher() -> bool:
    """Start the background rotation loop. No-op without a cookie file."""
    global _refresher_thread
    if not CONFIG.get("cookie_refresh_enabled", True):
        return False
    if not CONFIG.get("cookie_file"):
        return False
    if _refresher_thread is not None and _refresher_thread.is_alive():
        return True

    stop_event = threading.Event()
    _refresher_thread = threading.Thread(
        target=_refresh_loop, args=(stop_event,), name="cookie-refresh", daemon=True
    )
    _refresher_thread.start()
    return True
