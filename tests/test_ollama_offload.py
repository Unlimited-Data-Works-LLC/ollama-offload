#!/usr/bin/env python3
"""Offline tests for the Python twin.

No Ollama, no GPU, no network: every HTTP call is served from canned fixtures.
A test suite that needs a live inference host is a suite nobody runs.

    python3 -m unittest discover -s tests

The two tests that carry the library's central claim are
``test_non_json_200_does_not_arm_cooldown`` and
``test_transport_failure_arms_cooldown``. A 200 response whose content will not
parse is the MODEL failing; the endpoint is fine and must stay in service. Only
a transport failure justifies benching a host, because benching a healthy one
punishes every other caller that shares it.
"""

import importlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the object urlopen returns as a context manager.

    ``status`` matters: the reachability pre-flight checks ``resp.status == 200``
    and swallows every exception, so a fake without it reports the host as down
    rather than failing loudly.
    """

    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _routed_urlopen(routes, record=None):
    """Build a urlopen replacement that dispatches on URL suffix.

    A route value may be a dict/list (encoded as JSON), raw bytes, or an
    Exception instance, which is raised instead of returned.
    """

    def _urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if record is not None and getattr(req, "data", None):
            record.append(json.loads(req.data.decode("utf-8")))
        for suffix, value in routes.items():
            if url.endswith(suffix):
                if isinstance(value, Exception):
                    raise value
                if isinstance(value, bytes):
                    return _FakeResponse(value)
                return _FakeResponse(json.dumps(value).encode("utf-8"))
        raise urllib.error.URLError(f"no fixture route for {url}")

    return _urlopen


def _chat_response(content, *, eval_count=11, prompt_eval_count=22):
    return {
        "message": {"content": content},
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
    }


# /api/show reports `parameters` as a newline-delimited STRING, not an object.
# Reading it as though it had fields is the defect that broke the PowerShell
# twin, so the fixture keeps the real shape.
SHOW = {
    "parameters": "num_keep                       24\nstop                           \"<|im_end|>\"",
    "model_info": {"qwen3.context_length": 262144},
}
PS = {"models": [{"name": "qwen3:8b", "context_length": 8192}]}
TAGS = {"models": [{"name": "qwen3:8b"}]}
VERSION = {"version": "0.12.0"}

# Every endpoint a healthy call touches before /api/chat: the reachability
# pre-flight, model discovery, and context-window discovery.
HEALTHY = {"/api/version": VERSION, "/api/ps": PS, "/api/tags": TAGS, "/api/show": SHOW}

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


class OllamaOffloadTestCase(unittest.TestCase):
    """Reloads the module per test so module-scope caches never leak across cases."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = dict(os.environ)
        # Cooldown sidecars live under TEMP; redirect so tests never touch the
        # developer's real cooldown state, and never see each other's.
        os.environ["TEMP"] = self._tmp.name
        os.environ["TMP"] = self._tmp.name
        os.environ["OLLAMA_OFFLOAD_URL"] = "http://test.invalid:11434/api/chat"
        for noisy in ("OLLAMA_OFFLOAD_HOST", "OLLAMA_OFFLOAD_MODEL"):
            os.environ.pop(noisy, None)
        import ollama_offload

        self.mod = importlib.reload(ollama_offload)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        self._tmp.cleanup()

    def _install(self, routes, record=None):
        self.mod.urllib.request.urlopen = _routed_urlopen(routes, record)


class TestFaultSeparation(OllamaOffloadTestCase):
    def test_non_json_200_does_not_arm_cooldown(self):
        """A 200 carrying unparseable content is a MODEL fault, not a host fault."""
        self._install({**HEALTHY,
                       "/api/chat": _chat_response("I'm afraid I can't do that.")})
        with self.assertRaises(self.mod.OllamaCallError) as ctx:
            self.mod.call_ollama("go", schema=SCHEMA)
        # Specifically NOT the unavailable subclass -- the endpoint is healthy.
        self.assertNotIsInstance(ctx.exception, self.mod.OllamaUnavailable)
        in_cooldown, _ = self.mod._in_cooldown_for(self.mod.active_host_name())
        self.assertFalse(in_cooldown, "a model fault must never bench the endpoint")

    def test_transport_failure_arms_cooldown(self):
        """A connection failure IS a host fault, and must bench the host."""
        self.mod._BASE_BACKOFF_S = 0  # keep the retry loop instant
        self._install({**HEALTHY,
                       "/api/chat": urllib.error.URLError("connection refused")})
        with self.assertRaises(self.mod.OllamaUnavailable):
            self.mod.call_ollama("go", schema=SCHEMA)
        in_cooldown, remaining = self.mod._in_cooldown_for(self.mod.active_host_name())
        self.assertTrue(in_cooldown, "a transport fault must arm the cooldown")
        self.assertGreater(remaining, 0)


class TestShowParametersIsAString(OllamaOffloadTestCase):
    def test_context_discovery_survives_string_parameters(self):
        """/api/show returns `parameters` as a string; treating it as an object throws."""
        self._install(dict(HEALTHY))
        self.assertIsInstance(SHOW["parameters"], str)
        tokens = self.mod._discover_context_tokens("qwen3:8b")
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)


class TestWireBody(OllamaOffloadTestCase):
    def _call_and_capture(self):
        sent = []
        self._install({**HEALTHY,
                       "/api/chat": _chat_response(json.dumps({"ok": True}))}, record=sent)
        result = self.mod.call_ollama("go", schema=SCHEMA)
        chat_bodies = [b for b in sent if "messages" in b]
        return result, chat_bodies[-1]

    def test_options_carry_both_temperature_and_num_ctx(self):
        """Both knobs travel together; either one alone was a twin-drift bug."""
        _, body = self._call_and_capture()
        self.assertIn("temperature", body["options"])
        self.assertIn("num_ctx", body["options"])

    def test_keep_alive_is_sent_explicitly(self):
        """Omitting keep_alive lets the server apply its own default and evict the model."""
        _, body = self._call_and_capture()
        self.assertIn("keep_alive", body)

    def test_schema_is_sent_as_format(self):
        """Structured output rides on /api/chat's documented `format` field."""
        _, body = self._call_and_capture()
        self.assertEqual(body.get("format"), SCHEMA)

    def test_meta_keys(self):
        """The meta block the PowerShell twin must match, key for key."""
        result, _ = self._call_and_capture()
        self.assertEqual(
            set(result["_meta"]),
            {"model", "active_host", "elapsed_s", "eval_tokens", "prompt_tokens"},
        )
        self.assertEqual(result["_meta"]["eval_tokens"], 11)
        self.assertEqual(result["_meta"]["prompt_tokens"], 22)


class TestConfigRelocation(OllamaOffloadTestCase):
    def test_config_path_follows_the_env_var(self):
        """Vendoring this module read-only requires the config to live elsewhere.

        Without the override the config must sit inside the vendored copy, so
        every update overwrites local settings and a caller's real hosts cannot
        be kept outside the dependency.
        """
        elsewhere = os.path.join(self._tmp.name, "my_hosts.json")
        with open(elsewhere, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://elsewhere.invalid:11434/api/chat",
                       "model": "some-model",
                       "reviewer": {"context_usage_ratio": 0.25}}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = elsewhere
        os.environ.pop("OLLAMA_OFFLOAD_URL", None)
        import ollama_offload

        mod = importlib.reload(ollama_offload)
        self.assertEqual(str(mod._CONFIG_PATH), elsewhere)
        self.assertEqual(mod._CFG.get("model"), "some-model")
        # A consumer section from the relocated file must resolve, not fall
        # back to {} and silently take default ratios.
        self.assertEqual(mod.get_consumer_config("reviewer"),
                         {"context_usage_ratio": 0.25})


class TestNonCoincidentModelDiscovery(unittest.TestCase):
    """The stock fixture pins config model == /api/ps == qwen3:8b, so a bug that
    resolves the wrong side of the two would look identical to the fix. These
    cases pull the two names apart so only ONE order can pass each assertion.

    Two traps the DA pass on the first cut of this class found (aaf11bb → fix):
      1. ``importlib.reload`` runs ``_discover_model()`` at IMPORT — before any
         per-test urlopen patch is installed. Without patching first, the
         module's import-time discovery hits real urllib, resolves against a
         phony host (DNS fails), and falls back to ``_DEFAULT_MODEL_HINT``. If
         the fixture /api/ps happens to name the SAME string as the hint, the
         test passes via the hint — the exact coincidence it claims to exclude.
      2. Test order was also part of the leak: a sibling test's un-restored
         urlopen patch (or its absence) changed what the reload saw. Running
         ``TestFaultSeparation`` right before this class made the test FAIL,
         proving the pass was leak-dependent.
    Fix: install urlopen on ``urllib.request`` BEFORE the reload, AND pick a
    ``/api/ps`` model name that is NOT equal to ``_DEFAULT_MODEL_HINT`` so a
    hint firing cannot masquerade as a discovery hit.
    """

    # Deliberately not ``qwen3:8b`` or ``qwen3.6:35b`` (those are, respectively,
    # the pre-3987814 and post-3987814 values of ``_DEFAULT_MODEL_HINT`` — any
    # hint fall-through would coincide with one of them and pass the assertion
    # by accident). ``phony-model:test`` cannot possibly equal the hint.
    NONCOINCIDENT_MODEL = "phony-model:test"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = dict(os.environ)
        os.environ["TEMP"] = self._tmp.name
        os.environ["TMP"] = self._tmp.name
        os.environ.pop("OLLAMA_OFFLOAD_URL", None)
        for noisy in ("OLLAMA_OFFLOAD_HOST", "OLLAMA_OFFLOAD_MODEL"):
            os.environ.pop(noisy, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        self._tmp.cleanup()

    def _canned(self, ps_model):
        return {
            "/api/version": VERSION,
            "/api/ps": {"models": [{"name": ps_model, "context_length": 8192}]},
            "/api/tags": {"models": [{"name": ps_model}]},
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }

    def _reload_with_config_and_routes(self, cfg, ps_model):
        """Install the urlopen mock BEFORE ``importlib.reload`` so import-time
        ``_discover_model()`` probes the fixture rather than real urllib."""
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        sent: list = []
        urllib.request.urlopen = _routed_urlopen(self._canned(ps_model),
                                                 record=sent)
        import ollama_offload
        return importlib.reload(ollama_offload), sent

    def test_discovered_reaches_wire_when_config_omits_model(self):
        """No config pin: /api/ps says phony-model:test; body must send it.

        With the fixture pulled apart from ``_DEFAULT_MODEL_HINT``, only a
        genuine discovery hit can satisfy the assertion — a hint fall-through
        would send ``qwen3.6:35b`` and fail.
        """
        mod, sent = self._reload_with_config_and_routes(
            {"url": "http://test.invalid:11434/api/chat"},
            ps_model=self.NONCOINCIDENT_MODEL,
        )
        result = mod.call_ollama("go", schema=SCHEMA)
        chat_bodies = [b for b in sent if "messages" in b]
        self.assertEqual(chat_bodies[-1]["model"], self.NONCOINCIDENT_MODEL,
                         "with no config pin, the wire body must carry the "
                         "value discovered from /api/ps — not the hint fallback")
        self.assertEqual(result["_meta"]["model"], self.NONCOINCIDENT_MODEL)

    def test_config_pin_wins_over_discovered(self):
        """Config pin set to 'explicit-pin'; /api/ps disagrees. Pin must win."""
        mod, sent = self._reload_with_config_and_routes(
            {"url": "http://test.invalid:11434/api/chat", "model": "explicit-pin"},
            ps_model=self.NONCOINCIDENT_MODEL,
        )
        result = mod.call_ollama("go", schema=SCHEMA)
        chat_bodies = [b for b in sent if "messages" in b]
        self.assertEqual(chat_bodies[-1]["model"], "explicit-pin",
                         "a configured model pin must override any discovered "
                         "value on the wire")
        self.assertEqual(result["_meta"]["model"], "explicit-pin")

    def test_env_override_wins_over_config_pin(self):
        """``OLLAMA_OFFLOAD_MODEL`` per README:136 must override the pin.

        Without this test the F2 defect (pin winning over env) rides forever;
        the shipped ``ollama_offload_config.json`` pins ``model`` at top level
        AND per host, so an operator following the README today gets the env
        var silently ignored.
        """
        os.environ["OLLAMA_OFFLOAD_MODEL"] = "env-wins"
        mod, sent = self._reload_with_config_and_routes(
            {"url": "http://test.invalid:11434/api/chat", "model": "config-pin"},
            ps_model=self.NONCOINCIDENT_MODEL,
        )
        result = mod.call_ollama("go", schema=SCHEMA)
        chat_bodies = [b for b in sent if "messages" in b]
        self.assertEqual(chat_bodies[-1]["model"], "env-wins",
                         "OLLAMA_OFFLOAD_MODEL must win over a config pin per "
                         "README:136 — the shipped config pins, so this is the "
                         "only path the documented override actually reaches")
        self.assertEqual(result["_meta"]["model"], "env-wins")


class TestEnvironmentNamespace(unittest.TestCase):
    def test_ollama_host_is_not_read(self):
        """OLLAMA_HOST belongs to Ollama itself; reading it would hijack the user's setting."""
        src = os.path.join(SRC, "ollama_offload.py")
        with open(src, encoding="utf-8") as fh:
            body = fh.read()
        for stolen in ('"OLLAMA_HOST"', '"OLLAMA_MODELS"', '"OLLAMA_KEEP_ALIVE"'):
            self.assertNotIn(
                stolen, body,
                f"{stolen} is Ollama's own variable; this library must namespace to "
                f"OLLAMA_OFFLOAD_*",
            )


if __name__ == "__main__":
    unittest.main()
