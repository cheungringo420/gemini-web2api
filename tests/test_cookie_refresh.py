import errno
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from gemini_web2api import cookie_refresh
from gemini_web2api.config import CONFIG
from gemini_web2api.gemini import _cookie_cache


class FakeHeaders:
    """Mimics curl_cffi's multi-valued header mapping."""

    def __init__(self, pairs):
        self._pairs = pairs

    def multi_items(self):
        return list(self._pairs)


class FakeResponse:
    def __init__(self, status_code, header_pairs):
        self.status_code = status_code
        self.headers = FakeHeaders(header_pairs)


COOKIE_BASE = (
    "SID=sid-value; SAPISID=sapisid-value; "
    "__Secure-1PSID=1psid-value; __Secure-1PSIDTS=sidts-OLD"
)


class ParseSetCookieTests(unittest.TestCase):
    def test_reads_name_and_value_before_attributes(self):
        pairs = cookie_refresh.parse_set_cookie_pairs([
            "__Secure-1PSIDTS=sidts-NEW; Path=/; Secure; HttpOnly",
            "NID=534=abc; expires=Thu, 18-Feb-2027 06:24:28 GMT; path=/",
        ])
        self.assertEqual(pairs["__Secure-1PSIDTS"], "sidts-NEW")
        self.assertEqual(pairs["NID"], "534=abc")

    def test_ignores_deletion_and_malformed_lines(self):
        pairs = cookie_refresh.parse_set_cookie_pairs([
            "__Secure-1PSIDTS=; Max-Age=0",
            "no-equals-sign",
            "",
            None,
        ])
        self.assertEqual(pairs, {})


class MergeCookieStringTests(unittest.TestCase):
    def test_replaces_in_place_and_keeps_order(self):
        merged = cookie_refresh.merge_cookie_string(
            COOKIE_BASE, {"__Secure-1PSIDTS": "sidts-NEW"}
        )
        self.assertIn("__Secure-1PSIDTS=sidts-NEW", merged)
        self.assertNotIn("sidts-OLD", merged)
        self.assertTrue(merged.startswith("SID=sid-value"))
        self.assertIn("SAPISID=sapisid-value", merged)

    def test_appends_a_cookie_the_jar_did_not_have(self):
        merged = cookie_refresh.merge_cookie_string(
            COOKIE_BASE, {"__Secure-3PSIDTS": "sidts-3P"}
        )
        self.assertTrue(merged.endswith("__Secure-3PSIDTS=sidts-3P"))
        self.assertIn("__Secure-1PSIDTS=sidts-OLD", merged)

    def test_no_updates_leaves_the_jar_untouched(self):
        self.assertEqual(cookie_refresh.merge_cookie_string(COOKIE_BASE, {}), COOKIE_BASE)


class PersistRotatedCookiesTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        self.original_cache = dict(_cookie_cache)
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "gemini-auth.json")
        CONFIG["cookie_file"] = self.path

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)
        _cookie_cache.clear()
        _cookie_cache.update(self.original_cache)
        self.tempdir.cleanup()

    def write(self, payload):
        with open(self.path, "w") as handle:
            handle.write(payload)

    def test_updates_json_file_and_keeps_sibling_fields(self):
        self.write(json.dumps({"cookie": COOKIE_BASE, "sapisid": "sapisid-value"}))

        self.assertEqual(
            cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"}),
            "written",
        )

        with open(self.path) as handle:
            data = json.load(handle)
        self.assertIn("__Secure-1PSIDTS=sidts-NEW", data["cookie"])
        self.assertEqual(data["sapisid"], "sapisid-value")

    def test_supports_the_plain_cookie_string_file_format(self):
        self.write(COOKIE_BASE)

        self.assertEqual(
            cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"}),
            "written",
        )

        with open(self.path) as handle:
            content = handle.read()
        self.assertFalse(content.startswith("{"))
        self.assertIn("__Secure-1PSIDTS=sidts-NEW", content)

    def test_keeps_the_session_file_owner_only(self):
        self.write(json.dumps({"cookie": COOKIE_BASE}))
        os.chmod(self.path, 0o600)

        cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"})

        mode = stat.S_IMODE(os.stat(self.path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_refreshes_the_in_memory_cache_so_requests_use_the_new_value(self):
        self.write(json.dumps({"cookie": COOKIE_BASE, "sapisid": "sapisid-value"}))

        cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"})

        self.assertIn("sidts-NEW", _cookie_cache["str"])
        self.assertEqual(_cookie_cache["sapisid"], "sapisid-value")

    def test_re_reads_the_file_so_an_extension_write_is_not_clobbered(self):
        self.write(json.dumps({"cookie": COOKIE_BASE}))
        # The browser extension replaces the whole jar while we hold stale state.
        self.write(json.dumps({"cookie": "SID=sid-FROM-EXTENSION; __Secure-1PSIDTS=sidts-OLD"}))

        cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"})

        with open(self.path) as handle:
            cookie = json.load(handle)["cookie"]
        self.assertIn("SID=sid-FROM-EXTENSION", cookie)
        self.assertIn("__Secure-1PSIDTS=sidts-NEW", cookie)

    def test_no_write_when_the_value_did_not_change(self):
        self.write(json.dumps({"cookie": COOKIE_BASE}))
        before = os.path.getmtime(self.path)

        self.assertEqual(
            cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-OLD"}),
            "unchanged",
        )
        self.assertEqual(os.path.getmtime(self.path), before)

    def test_leaves_no_temp_file_behind(self):
        self.write(json.dumps({"cookie": COOKIE_BASE}))
        cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"})
        self.assertEqual(os.listdir(self.tempdir.name), ["gemini-auth.json"])

    def test_bind_mounted_file_falls_back_to_an_in_place_rewrite(self):
        """os.replace() raises EBUSY on a bind mount -- the Docker default layout."""
        self.write(json.dumps({"cookie": COOKIE_BASE}))
        inode_before = os.stat(self.path).st_ino

        real_replace = os.replace

        def busy_replace(src, dst):
            if os.path.abspath(dst) == os.path.abspath(self.path):
                raise OSError(errno.EBUSY, "Device or resource busy")
            return real_replace(src, dst)

        with mock.patch("os.replace", side_effect=busy_replace):
            status = cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"})

        self.assertEqual(status, "written")
        with open(self.path) as handle:
            self.assertIn("sidts-NEW", json.load(handle)["cookie"])
        # Rewritten in place, so the inode the bind mount points at is preserved.
        self.assertEqual(os.stat(self.path).st_ino, inode_before)
        self.assertEqual(os.listdir(self.tempdir.name), ["gemini-auth.json"])

    def test_read_only_file_reports_write_failed_and_leaves_no_copy(self):
        """A read-only mount must not look like "nothing to do"."""
        self.write(json.dumps({"cookie": COOKIE_BASE}))

        def readonly_replace(src, dst):
            raise OSError(errno.EBUSY, "Device or resource busy")

        original_open = open

        def guarded_open(path, mode="r", *args, **kwargs):
            if os.path.abspath(str(path)) == os.path.abspath(self.path) and "r+" in mode:
                raise OSError(errno.EROFS, "Read-only file system")
            return original_open(path, mode, *args, **kwargs)

        with mock.patch("os.replace", side_effect=readonly_replace), \
                mock.patch("builtins.open", side_effect=guarded_open):
            status = cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"})

        self.assertEqual(status, "write_failed")
        with open(self.path) as handle:
            self.assertIn("sidts-OLD", json.load(handle)["cookie"])
        self.assertEqual(os.listdir(self.tempdir.name), ["gemini-auth.json"])

    def test_missing_cookie_file_is_not_created(self):
        CONFIG["cookie_file"] = os.path.join(self.tempdir.name, "absent.json")
        self.assertEqual(
            cookie_refresh.persist_rotated_cookies({"__Secure-1PSIDTS": "sidts-NEW"}),
            "no_file",
        )
        self.assertFalse(os.path.exists(CONFIG["cookie_file"]))


class RotateCookiesOnceTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        self.original_cache = dict(_cookie_cache)
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "gemini-auth.json")
        with open(self.path, "w") as handle:
            handle.write(json.dumps({"cookie": COOKIE_BASE, "sapisid": "sapisid-value"}))
        CONFIG["cookie_file"] = self.path
        CONFIG["proxy"] = None
        _cookie_cache.update({"str": "", "sapisid": None, "mtime": 0})

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)
        _cookie_cache.clear()
        _cookie_cache.update(self.original_cache)
        self.tempdir.cleanup()

    def test_persists_the_rotated_cookie_on_success(self):
        response = FakeResponse(200, [
            ("set-cookie", "__Secure-1PSIDTS=sidts-NEW; Path=/; Secure"),
            ("content-type", "application/json"),
        ])
        with mock.patch.object(cookie_refresh, "HAS_CURL_CFFI", True), \
                mock.patch.object(cookie_refresh, "curl_requests") as requests:
            requests.post.return_value = response
            ok, detail = cookie_refresh.rotate_cookies_once()

        self.assertTrue(ok)
        self.assertEqual(detail, "__Secure-1PSIDTS")
        with open(self.path) as handle:
            self.assertIn("sidts-NEW", json.load(handle)["cookie"])

    def test_reports_a_signed_out_session_without_touching_the_file(self):
        response = FakeResponse(401, [])
        with mock.patch.object(cookie_refresh, "HAS_CURL_CFFI", True), \
                mock.patch.object(cookie_refresh, "curl_requests") as requests:
            requests.post.return_value = response
            ok, detail = cookie_refresh.rotate_cookies_once()

        self.assertFalse(ok)
        self.assertEqual(detail, "unauthorized")
        with open(self.path) as handle:
            self.assertIn("sidts-OLD", json.load(handle)["cookie"])

    def test_a_200_without_a_rotating_cookie_is_not_treated_as_success(self):
        response = FakeResponse(200, [("set-cookie", "NID=534=abc; path=/")])
        with mock.patch.object(cookie_refresh, "HAS_CURL_CFFI", True), \
                mock.patch.object(cookie_refresh, "curl_requests") as requests:
            requests.post.return_value = response
            ok, detail = cookie_refresh.rotate_cookies_once()

        self.assertFalse(ok)
        self.assertEqual(detail, "no_rotating_cookie")

    def test_network_failure_is_reported_not_raised(self):
        with mock.patch.object(cookie_refresh, "HAS_CURL_CFFI", True), \
                mock.patch.object(cookie_refresh, "curl_requests") as requests:
            requests.post.side_effect = OSError("connection reset")
            ok, detail = cookie_refresh.rotate_cookies_once()

        self.assertFalse(ok)
        self.assertEqual(detail, "error:OSError")

    def test_skips_entirely_without_a_cookie(self):
        CONFIG["cookie_file"] = None
        _cookie_cache.update({"str": "", "sapisid": None, "mtime": 0})
        self.assertEqual(cookie_refresh.rotate_cookies_once(), (False, "no_cookie"))


class RefresherStartupTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_does_not_start_without_a_cookie_file(self):
        CONFIG["cookie_file"] = None
        self.assertFalse(cookie_refresh.start_cookie_refresher())

    def test_respects_the_disable_switch(self):
        CONFIG["cookie_file"] = "/tmp/whatever.json"
        CONFIG["cookie_refresh_enabled"] = False
        self.assertFalse(cookie_refresh.start_cookie_refresher())

    def test_backoff_is_capped_so_a_429_cannot_outlast_the_cookie(self):
        """A skipped refresh is a session that quietly expires -- do not wait forever."""
        CONFIG["cookie_refresh_interval_sec"] = 1800
        interval = cookie_refresh.refresh_interval_sec()
        backoff = min(interval * cookie_refresh.FAILURE_BACKOFF_MULTIPLIER,
                      cookie_refresh.MAX_BACKOFF_SEC)
        self.assertLessEqual(backoff, cookie_refresh.MAX_BACKOFF_SEC)
        self.assertEqual(backoff, 3600)

    def test_default_cadence_leaves_room_under_the_rate_limit(self):
        CONFIG.pop("cookie_refresh_interval_sec", None)
        self.assertEqual(cookie_refresh.refresh_interval_sec(), 1800)

    def test_interval_floor_keeps_the_loop_from_hammering(self):
        CONFIG["cookie_refresh_interval_sec"] = 1
        self.assertEqual(cookie_refresh.refresh_interval_sec(), 60)

    def test_interval_falls_back_when_misconfigured(self):
        CONFIG["cookie_refresh_interval_sec"] = "not-a-number"
        self.assertEqual(
            cookie_refresh.refresh_interval_sec(),
            cookie_refresh.DEFAULT_REFRESH_INTERVAL_SEC,
        )


if __name__ == "__main__":
    unittest.main()
