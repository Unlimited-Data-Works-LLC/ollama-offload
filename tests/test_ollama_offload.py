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
import threading
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

    def test_hint_never_reaches_wire_when_discovery_fails_at_import(self):
        """F5: primary DOWN at import → hint fired → call_ollama must NOT
        ship the hint on the wire; a fresh discovery is attempted, and if
        that also fails, OllamaCallError is raised. The library's central
        fault-separation claim depends on this — benching a healthy host on
        a wrong-model 4xx would invert it (see :493-499).
        """
        # Set up: urlopen that fails BOTH /api/ps and /api/tags — the two
        # sources _discover_model consults. /api/version + /api/chat still
        # answer so the reachability pre-flight + the actual call succeed
        # up to the resolution site.
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down"),
            "/api/tags": urllib.error.URLError("primary down"),
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        # Confirm the hint fired at import.
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "hint")
        # Now attempt a call; the guard must intercept before shipping the
        # hint to any host.
        with self.assertRaises(mod.OllamaCallError) as ctx:
            mod.call_ollama("go", schema=SCHEMA)
        self.assertIn("_DEFAULT_MODEL_HINT", str(ctx.exception),
                      "the error must name the discovery-fallback so an "
                      "operator can act on it, not surface as a generic call "
                      "failure")

    def test_env_value_stripped_before_wire(self):
        """DA-round2 (iv): whitespace-bearing OLLAMA_OFFLOAD_MODEL must be
        stripped, or Ollama 404s on `"  env-wins  "`."""
        os.environ["OLLAMA_OFFLOAD_MODEL"] = "  env-wins  "
        mod, sent = self._reload_with_config_and_routes(
            {"url": "http://test.invalid:11434/api/chat", "model": "config-pin"},
            ps_model=self.NONCOINCIDENT_MODEL,
        )
        mod.call_ollama("go", schema=SCHEMA)
        chat_bodies = [b for b in sent if "messages" in b]
        self.assertEqual(chat_bodies[-1]["model"], "env-wins",
                         "OLLAMA_OFFLOAD_MODEL must be stripped on read before "
                         "shipping on the wire")

    def test_empty_string_env_is_unset_not_win(self):
        """DA-round2 (MISSED): OLLAMA_OFFLOAD_MODEL='' must fall through to
        the config pin. A future refactor from `if override:` to
        `if override is not None:` would silently invert this — the test
        locks the intended behaviour."""
        os.environ["OLLAMA_OFFLOAD_MODEL"] = ""
        mod, sent = self._reload_with_config_and_routes(
            {"url": "http://test.invalid:11434/api/chat", "model": "config-pin"},
            ps_model=self.NONCOINCIDENT_MODEL,
        )
        mod.call_ollama("go", schema=SCHEMA)
        chat_bodies = [b for b in sent if "messages" in b]
        self.assertEqual(chat_bodies[-1]["model"], "config-pin",
                         "empty-string env is UNSET; the config pin must win")

    def test_default_model_refreshed_after_import_hint_fallback(self):
        """DA-round2 (ii CRITICAL): sticky-hint regression. Import with
        primary DOWN → hint fires. Primary recovers → call 1's F5 retry
        finds the real model → wire body carries it. Call 2 must ALSO
        carry the recovered model, not the hint via the else-branch —
        proving DEFAULT_MODEL was refreshed on F5 success."""
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down at import"),
            "/api/tags": urllib.error.URLError("primary down at import"),
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "hint",
                         "precondition: import-time discovery must have hit "
                         "the hint fallback")
        self.assertEqual(mod.DEFAULT_MODEL, mod._DEFAULT_MODEL_HINT,
                         "precondition: module DEFAULT_MODEL holds the hint")
        # Primary recovers between import and call: swap routes to healthy.
        sent: list = []
        mod.urllib.request.urlopen = _routed_urlopen(
            self._canned(self.NONCOINCIDENT_MODEL), record=sent)
        # Call 1 — F5 retry probes, finds phony-model:test, ships it.
        mod.call_ollama("go", schema=SCHEMA)
        chat1 = [b for b in sent if "messages" in b][-1]
        self.assertEqual(chat1["model"], self.NONCOINCIDENT_MODEL,
                         "call 1 must ship the recovered model, not the hint")
        # Call 2 — _DEFAULT_MODEL_SOURCE is no longer 'hint' so the F5 branch
        # is skipped; execution falls to `_effective_model = DEFAULT_MODEL`.
        # Without the DEFAULT_MODEL refresh, this ships the stale hint.
        sent.clear()
        mod.urllib.request.urlopen = _routed_urlopen(
            self._canned(self.NONCOINCIDENT_MODEL), record=sent)
        mod.call_ollama("go", schema=SCHEMA)
        chat2 = [b for b in sent if "messages" in b][-1]
        self.assertEqual(chat2["model"], self.NONCOINCIDENT_MODEL,
                         "call 2 must ALSO ship the recovered model — proves "
                         "DEFAULT_MODEL was refreshed on F5 retry success")
        self.assertNotEqual(chat2["model"], mod._DEFAULT_MODEL_HINT,
                            "call 2 must NOT ship the sticky hint")

    def test_whitespace_only_env_falls_through_to_config_pin(self):
        """DA-round3 (B) + DA-round4 (C): whitespace-only OLLAMA_OFFLOAD_MODEL
        must fall through to the config pin, not ship as the wire model. The
        `.strip()` at :751 (and :378) collapses it to '' -> falsy -> pin
        wins. Empty-string is already locked by
        `test_empty_string_env_is_unset_not_win`; whitespace-only is a
        distinct input class that a refactor removing either `.strip()`
        call would silently ship as-is -> 404 on the wire.

        DA-round4 (C) parametrized this beyond the original ``'   '`` case:
        the whitespace category is not one input but a family (tab,
        newline, NBSP, mixed), and a `.strip()` call that drops any of them
        from its default set (Python's `str.strip()` covers ASCII whitespace
        AND NBSP `\\u00a0` because it treats characters `.isspace()` == True)
        must ALSO drop the rest. A refactor to a hand-rolled strip on
        `" \\t\\n"` alone would leak NBSP; parametrization catches that.
        """
        for label, value in (
            ("spaces", "   "),
            ("tab", "\t"),
            ("newline", "\n"),
            ("nbsp", " "),
            ("mixed", " \t \n "),
        ):
            with self.subTest(whitespace=label):
                # Fresh env + module reload per sub-case so the module-scope
                # `_DEFAULT_MODEL_SOURCE` state doesn't leak between cases.
                # DA-round5 (cleanup): wrap the mutation in try/finally so
                # the env pop runs even if an assertion raises. Without it,
                # a failure in one sub-case leaks OLLAMA_OFFLOAD_MODEL into
                # the next — that would show up as a *different* failure
                # class than the real one, misleading the debugger.
                os.environ["OLLAMA_OFFLOAD_MODEL"] = value
                try:
                    mod, sent = self._reload_with_config_and_routes(
                        {"url": "http://test.invalid:11434/api/chat",
                         "model": "config-pin"},
                        ps_model=self.NONCOINCIDENT_MODEL,
                    )
                    mod.call_ollama("go", schema=SCHEMA)
                    chat_bodies = [b for b in sent if "messages" in b]
                    self.assertEqual(
                        chat_bodies[-1]["model"], "config-pin",
                        f"whitespace-only env ({label!r}) is UNSET after "
                        f"strip; the config pin must win — the wire body "
                        f"must NOT ship {value!r}")
                finally:
                    os.environ.pop("OLLAMA_OFFLOAD_MODEL", None)

    def test_discover_model_strips_env_override_directly(self):
        """DA-round3 (C): the `.strip()` at :378 in `_discover_model` is
        not directly test-locked — the end-to-end wire tests all traverse
        `_call_ollama_once` at :751, which strips independently. If a
        refactor removes :378's strip while :751 stays, wire bodies stay
        clean but `_DEFAULT_MODEL_SOURCE == "env"` after a call to
        `_discover_model` returns the UNSTRIPPED value — silent drift for
        any consumer that reads DEFAULT_MODEL / calls _discover_model."""
        os.environ["OLLAMA_OFFLOAD_MODEL"] = "  env-wins  "
        mod, _sent = self._reload_with_config_and_routes(
            {"url": "http://test.invalid:11434/api/chat", "model": "config-pin"},
            ps_model=self.NONCOINCIDENT_MODEL,
        )
        self.assertEqual(mod._discover_model(), "env-wins",
                         "_discover_model must strip the env override on read")
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "env")

    def test_default_context_tokens_refreshed_after_import_hint_fallback(self):
        """DA-round3 (A): post-c0961a3 the num_ctx shortcut at :818 fires
        on every subsequent same-host call, because the F5 branch above
        rewrites DEFAULT_MODEL to equal _effective_model on retry success
        -- the guard `_effective_model == DEFAULT_MODEL` is TRUE by
        construction from call 2 onward. If DEFAULT_CONTEXT_TOKENS is not
        ALSO refreshed on that F5 success, the shortcut ships the
        IMPORT-TIME context window (discovered against the HINT model,
        possibly the 8192 fallback or an architectural /api/show value
        for a model that isn't even loaded on the recovered host) on
        every wire body from call 2 forward -- silent truncation or
        over-fill against the real serving model."""
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        # Same import-time-dead pattern as
        # `test_default_model_refreshed_after_import_hint_fallback`:
        # /api/ps + /api/tags fail so _discover_model hits the hint;
        # /api/show still answers so _discover_context_tokens at import
        # returns the architectural value from model_info (262144 from
        # the SHOW fixture's qwen3.context_length).
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down at import"),
            "/api/tags": urllib.error.URLError("primary down at import"),
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "hint",
                         "precondition: import-time discovery hit the hint")
        self.assertEqual(mod.DEFAULT_CONTEXT_TOKENS, 262144,
                         "precondition: import-time context is the "
                         "architectural /api/show value -- WRONG for the "
                         "recovered model's actual loaded num_ctx")
        # Primary recovers between import and call: /api/ps returns
        # phony-model:test @ context_length=8192. The recovered model's
        # runtime context (8192) MUST reach the wire on BOTH calls, not
        # the stale architectural 262144.
        sent: list = []
        mod.urllib.request.urlopen = _routed_urlopen(
            self._canned(self.NONCOINCIDENT_MODEL), record=sent)
        mod.call_ollama("go", schema=SCHEMA)
        chat1 = [b for b in sent if "messages" in b][-1]
        self.assertEqual(chat1["options"]["num_ctx"], 8192,
                         "call 1 must ship the recovered model's num_ctx, "
                         "not the import-time architectural value")
        # Call 2 -- _DEFAULT_MODEL_SOURCE is no longer 'hint' so the F5
        # branch is skipped; the num_ctx shortcut at :818 fires
        # (`_effective_model == DEFAULT_MODEL and _call_host is
        # _ACTIVE_HOST` both True by construction). Without the
        # DEFAULT_CONTEXT_TOKENS refresh, this ships the stale 262144.
        sent.clear()
        mod.urllib.request.urlopen = _routed_urlopen(
            self._canned(self.NONCOINCIDENT_MODEL), record=sent)
        mod.call_ollama("go", schema=SCHEMA)
        chat2 = [b for b in sent if "messages" in b][-1]
        self.assertEqual(chat2["options"]["num_ctx"], 8192,
                         "call 2 must ALSO ship the recovered num_ctx -- "
                         "proves DEFAULT_CONTEXT_TOKENS was refreshed on "
                         "F5 recovery; the c0961a3 shortcut at :818 would "
                         "otherwise ship the stale import-time value")
        self.assertNotEqual(chat2["options"]["num_ctx"], 262144,
                            "call 2 must NOT ship the stale architectural "
                            "value from the hint-era /api/show")

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

    def test_fail_open_probe_does_not_clobber_import_time_default_context(self):
        """DA-round4 (A): the F5 refresh must NOT clobber a
        possibly-more-accurate import-time DEFAULT_CONTEXT_TOKENS with the
        fallback CONSTANT when the recovery probe transiently fails. Setup:
        primary DOWN at import EXCEPT /api/show answers (so import-time
        DEFAULT_CONTEXT_TOKENS = 262144 from SHOW's model_info). Recovery:
        /api/tags succeeds (so `_discover_model` finds the real model), but
        /api/ps has NO models AND /api/show throws — `_discover_context_tokens`
        would return the fallback 8192. Assert DEFAULT_CONTEXT_TOKENS RETAINS
        262144; a plain `DEFAULT_CONTEXT_TOKENS = _discover_context_tokens(...)`
        writes 8192 and this assertion fails.
        """
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        # Import-time: /api/ps + /api/tags fail (hint fires for the model),
        # /api/show still answers (context = 262144 from model_info).
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down at import"),
            "/api/tags": urllib.error.URLError("primary down at import"),
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "hint",
                         "precondition: import hit the hint fallback")
        self.assertEqual(mod.DEFAULT_CONTEXT_TOKENS, 262144,
                         "precondition: import-time context is the "
                         "architectural /api/show value")
        # Recovery: /api/tags names the real model so _discover_model
        # succeeds; /api/ps returns EMPTY (model not loaded yet) and
        # /api/show throws — the strict context probe must return None
        # so the F5 write is skipped.
        sent: list = []
        recovery = {
            "/api/version": VERSION,
            "/api/ps": {"models": []},
            "/api/tags": {"models": [{"name": self.NONCOINCIDENT_MODEL}]},
            "/api/show": urllib.error.HTTPError(
                "http://x", 500, "Internal Server Error", {}, None),
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        mod.urllib.request.urlopen = _routed_urlopen(recovery, record=sent)
        pre_model = mod.DEFAULT_MODEL
        mod.call_ollama("go", schema=SCHEMA)
        # DA-round6 (residual): under all-or-nothing atomicity, a
        # strict-None from the probe skips BOTH writes — model AND
        # context — preserving whatever coherent pair the module
        # already held. Prior shape wrote MODEL and skipped CTX,
        # producing a cross-model pair with any prior writer's CTX
        # (DA-r6 finding 4). Property: on strict-None, no advance.
        self.assertEqual(mod.DEFAULT_MODEL, pre_model,
                         "on strict-None from the context probe the "
                         "F5 refresh must SKIP BOTH writes — MODEL "
                         "advancing while CTX stays is precisely the "
                         "DA-round6 cross-model pair bug")
        self.assertEqual(mod.DEFAULT_CONTEXT_TOKENS, 262144,
                         "DEFAULT_CONTEXT_TOKENS must retain the import-time "
                         "value (262144) — a fail-open probe returning the "
                         "fallback constant 8192 would clobber a possibly "
                         "more-accurate import-time value; the strict-None "
                         "guard skips the write")
        self.assertNotEqual(
            mod.DEFAULT_CONTEXT_TOKENS,
            int(mod._CFG.get("context_tokens_fallback", 8192)),
            "post-refresh value must NOT be the fallback constant")

    def test_context_bytes_transitions_atomically_across_f5_refresh(self):
        """DA-round4 (D): `context_bytes(ratio)` reads DEFAULT_CONTEXT_TOKENS
        lock-free. Under Fix B the F5 refresh commits DEFAULT_MODEL +
        DEFAULT_CONTEXT_TOKENS together under `_F5_REFRESH_LOCK`, so
        successive reads observe (old, old) or (new, new) — never a torn
        (new_model, old_tokens). Snapshot `context_bytes(0.5)` at THREE
        points: post-import (hint-era 262144), post-first-call (F5 flip
        to the recovered 98304), post-second-call (stable at 98304 — F5
        branch is skipped because source is no longer "hint"). Assert the
        transition happened once and later reads are consistent.
        """
        # Recovery ctx MUST NOT equal `_CFG['context_tokens_fallback']`
        # (default 8192) so the Fix-A strict-None guard would NOT skip
        # the write — this test measures the SUCCESS path, not the
        # fall-through path.
        RECOVERED_CTX = 98304
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down at import"),
            "/api/tags": urllib.error.URLError("primary down at import"),
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        # Point 1: post-import — hint-era 262144 from SHOW.
        cb_import = mod.context_bytes(0.5)
        self.assertEqual(cb_import, int(262144 * 0.5 * 4),
                         "point 1: post-import context_bytes reflects the "
                         "hint-era DEFAULT_CONTEXT_TOKENS (262144)")
        # Swap to a healthy recovery with a DIFFERENT context.
        recovery = {
            "/api/version": VERSION,
            "/api/ps": {"models": [
                {"name": self.NONCOINCIDENT_MODEL,
                 "context_length": RECOVERED_CTX}
            ]},
            "/api/tags": {"models": [{"name": self.NONCOINCIDENT_MODEL}]},
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        mod.urllib.request.urlopen = _routed_urlopen(recovery)
        mod.call_ollama("go", schema=SCHEMA)
        # Point 2: post-first-call — F5 flipped the pair together.
        cb_after_flip = mod.context_bytes(0.5)
        self.assertEqual(cb_after_flip, int(RECOVERED_CTX * 0.5 * 4),
                         "point 2: post-first-call context_bytes reflects "
                         "the F5-refreshed value — the flip must be "
                         "OBSERVABLE by the same reader that saw the "
                         "hint-era value")
        self.assertNotEqual(cb_after_flip, cb_import,
                            "the transition must have happened — otherwise "
                            "the refresh path is inert")
        # Second call: F5 branch skipped now (source != "hint"); value
        # must remain stable.
        mod.call_ollama("go", schema=SCHEMA)
        # Point 3: post-second-call — no further refresh, value stable.
        cb_stable = mod.context_bytes(0.5)
        self.assertEqual(cb_stable, cb_after_flip,
                         "point 3: post-second-call context_bytes must "
                         "match point 2 — no further F5 refresh runs "
                         "because _DEFAULT_MODEL_SOURCE is no longer "
                         "'hint'; readers see a stable value across the "
                         "lock boundary")

    def test_f5_strict_probe_runs_under_the_refresh_lock(self):
        """DA-round5 (composition): Fix-A (strict-None skips the ctx
        write) and Fix-B (commit the pair under `_F5_REFRESH_LOCK`)
        COMPOSE into a cross-model race unless the strict probe runs
        INSIDE the lock. Repro under a probe-outside-lock shape:

          - Writer B probes ctx=V for model B strict → V.
            Acquires the lock, writes (DEFAULT_MODEL=B,
            DEFAULT_CONTEXT_TOKENS=V), releases.
          - Writer A had probed ctx for model A concurrently,
            /api/show 500 → strict returns None.  Acquires the lock,
            writes DEFAULT_MODEL=A, SKIPS the ctx write per Fix A.
          - Final pair: (DEFAULT_MODEL=A, DEFAULT_CONTEXT_TOKENS=V-
            for-B) — a cross-model pair, exactly what the lock was
            supposed to prevent.

        Prior atomicity test at :643 is an ISOLATION test — no
        threading, asserts only single-value equality across a
        sequential three-point read. Delete `with _F5_REFRESH_LOCK:`
        from source entirely and prior Test D still passes.  That
        makes it a rubber-stamp for the atomic-pair claim; this test
        is the anti-rubber-stamp.

        Design: two threads enter `call_ollama` while
        `_DEFAULT_MODEL_SOURCE == "hint"`, forced concurrent via a
        `threading.Barrier(2)` inside a monkey-patched
        `_probe_context_tokens_strict`. The patched probe records the
        state of `_F5_REFRESH_LOCK.locked()` on every entry. Property:
        every probe entry must observe the lock as held. A probe
        outside the lock — or a lock removed entirely — makes the
        recorded value False and the assertion fails.

        Why the barrier: it forces two threads to be simultaneously
        inside the probe. Under the fix, only one thread at a time
        can be inside the probe (the lock serialises them), so a
        barrier of size 2 inside the probe would deadlock. The test
        therefore uses `Barrier(2, timeout=…)` and treats a barrier
        timeout on BOTH probes as the SUCCESS signal — probes are
        serialised, which is the fix's guarantee. Under old code
        (probe outside lock) both probes reach the barrier
        concurrently and the barrier releases; the recorded
        lock-held flag is False and the follow-up assertion fires.
        Two distinct signals, both anti-rubber-stamp:
          (a) probe under lock → barrier times out on both probes
              AND recorded lock-held is True on both,
          (b) probe outside lock → barrier releases AND recorded
              lock-held is False on at least one.
        """
        RECOVERED_CTX = 98304
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        # Import with primary DOWN so _DEFAULT_MODEL_SOURCE == "hint".
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down at import"),
            "/api/tags": urllib.error.URLError("primary down at import"),
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "hint",
                         "precondition: import must have hit the hint "
                         "fallback so both threads take the F5 branch")
        # Swap to a healthy recovery whose ctx != fallback constant, so
        # the Fix-A None-skip does NOT gate the ctx write on the SUCCESS
        # path — we're measuring the LOCK's atomicity claim here, not
        # the None sentinel.
        recovery = {
            "/api/version": VERSION,
            "/api/ps": {"models": [
                {"name": self.NONCOINCIDENT_MODEL,
                 "context_length": RECOVERED_CTX}
            ]},
            "/api/tags": {"models": [{"name": self.NONCOINCIDENT_MODEL}]},
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        mod.urllib.request.urlopen = _routed_urlopen(recovery)
        # Instrument the strict probe. Two threads race into the F5
        # block; the barrier waits for BOTH to reach the probe. Under
        # the fix, they can't — the second thread blocks on the F5
        # lock outside the probe. Under old code, both reach it.
        real_probe = mod._probe_context_tokens_strict
        barrier = threading.Barrier(2, timeout=1.5)
        entries = []
        entries_lock = threading.Lock()

        def instrumented(model, *, base_url=None):
            held = mod._F5_REFRESH_LOCK.locked()
            barrier_reached = True
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                # Expected under the fix: probes are serialised so a
                # 2-way barrier times out. That is the SUCCESS shape
                # for lock-held atomicity.
                barrier_reached = False
            with entries_lock:
                entries.append({
                    "thread": threading.current_thread().name,
                    "lock_held_at_probe": held,
                    "barrier_reached": barrier_reached,
                })
            return real_probe(model, base_url=base_url)

        mod._probe_context_tokens_strict = instrumented

        errors = []

        def worker():
            try:
                mod.call_ollama("go", schema=SCHEMA)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, name=f"F5-writer-{i}")
            for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
        for t in threads:
            self.assertFalse(t.is_alive(),
                             f"worker {t.name} did not finish — the "
                             f"F5 refresh deadlocked or hung")
        self.assertEqual(errors, [],
                         f"workers raised: {errors!r}")
        # Under the fix, at most ONE thread hits the F5 refresh: the
        # first thread through flips `_DEFAULT_MODEL_SOURCE` off "hint"
        # while holding the lock; the second thread's read of the
        # source at :812 sees the flipped value and skips F5 entirely.
        # That is itself an atomicity property: the source-flip and
        # the (model, ctx) commit are all under the same lock, so a
        # concurrent observer NEVER sees a half-applied refresh.
        # Under old code, the flip happens BEFORE the lock is taken
        # (via `_discover_model` at :822), both threads can pass the
        # :812 gate before either commits, and both reach the probe.
        # So the test discriminates on the count of probe entries too:
        # <=1 entry means the source-flip and commit were coupled
        # under a lock; ==2 entries means they were separable.
        self.assertGreaterEqual(len(entries), 1,
                                "the F5 strict probe never fired — the "
                                "F5 branch was not exercised by this "
                                "test (routes/mocks misconfigured)")
        for entry in entries:
            self.assertTrue(
                entry["lock_held_at_probe"],
                f"_probe_context_tokens_strict ran with "
                f"_F5_REFRESH_LOCK NOT held (thread "
                f"{entry['thread']!r}): probe-outside-lock reopens "
                f"the DA-round5 cross-model composition race. "
                f"entries={entries!r}")

    def test_f5_writer_pair_coherence_under_stubborn_none(self):
        """DA-round6 (residual): probe-inside-lock is necessary but not
        sufficient. Under the r5 shape (writes MODEL unconditionally,
        skips only CTX on strict-None), a stubborn transient inside the
        lock still corrupts the pair:

          - Writer B enters lock, probe returns V, writes
            (DEFAULT_MODEL=B, DEFAULT_CONTEXT_TOKENS=V), releases.
          - Writer A enters lock, probe STILL returns None (persistent
            /api/show 500, model reload in progress), writes
            DEFAULT_MODEL=A, SKIPS ctx write.
          - Final pair: (DEFAULT_MODEL=A, DEFAULT_CONTEXT_TOKENS=V-for-B)
            — cross-model, r5's property violated inside the lock.

        Fix: atomic all-or-nothing. On strict-None, skip BOTH writes so
        the pair either advances coherently to (B, V) or stays at the
        pre-state. Never a half-applied pair.

        Design: force B-first ordering so the bug shape is deterministic
        (A-first coincidentally produces a coherent pair under either
        code path). B's `_discover_model` returns MODEL_B, its probe
        returns V and signals an event. A's `_discover_model` waits on
        that event before returning MODEL_A, so A blocks on the F5 lock
        until B has committed. Then A's probe returns None, exercising
        the residual defect.
        """
        MODEL_B = "MODEL_B:test"
        MODEL_A = "MODEL_A:test"
        V = 98304
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down at import"),
            "/api/tags": urllib.error.URLError("primary down at import"),
            "/api/show": SHOW,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "hint",
                         "precondition: import must have hit hint")
        HINT_MODEL = mod.DEFAULT_MODEL
        HINT_CTX = mod.DEFAULT_CONTEXT_TOKENS
        self.assertNotEqual(HINT_MODEL, MODEL_B,
                            "test setup: hint model must differ from B")
        self.assertNotEqual(HINT_MODEL, MODEL_A,
                            "test setup: hint model must differ from A")

        # Feed /api/chat something callable; the F5 refresh short-circuits
        # before /api/chat under our mocks, but call_ollama still resolves
        # the effective host.
        mod.urllib.request.urlopen = _routed_urlopen({
            "/api/version": VERSION,
            "/api/chat": _chat_response(json.dumps({"ok": True})),
        })

        b_reached_discover = threading.Event()
        b_in_probe = threading.Event()

        def instrumented_discover(*, base_url=None):
            # Guarantee both threads pass the source=='hint' gate at
            # :812 before either flips source: use B's arrival as the
            # rendezvous — A's discover blocks until B is INSIDE its
            # probe (i.e. B holds the F5 lock and is already past the
            # source-flip). Then A flips (idempotent) and proceeds; A
            # will block on the F5 lock until B commits and releases.
            tname = threading.current_thread().name
            if "B" in tname:
                mod._DEFAULT_MODEL_SOURCE = "ps"
                b_reached_discover.set()
                return MODEL_B
            # A path
            self.assertTrue(b_reached_discover.wait(timeout=5.0),
                            "B never reached _discover_model")
            self.assertTrue(b_in_probe.wait(timeout=5.0),
                            "B never entered its probe")
            mod._DEFAULT_MODEL_SOURCE = "ps"
            return MODEL_A

        def instrumented_probe(model, *, base_url=None):
            # MODEL_B → return V (probe success). MODEL_A → return None
            # (stubborn transient inside the lock). Both branches run
            # while _F5_REFRESH_LOCK is held by their respective caller.
            if model == MODEL_B:
                b_in_probe.set()
                return V
            return None

        mod._discover_model = instrumented_discover
        mod._probe_context_tokens_strict = instrumented_probe

        errors = []

        def worker():
            try:
                mod.call_ollama("go", schema=SCHEMA)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, name="writer-B"),
            threading.Thread(target=worker, name="writer-A"),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
        for t in threads:
            self.assertFalse(t.is_alive(),
                             f"worker {t.name} deadlocked or hung")
        self.assertEqual(errors, [],
                         f"workers raised: {errors!r}")

        final_model = mod.DEFAULT_MODEL
        final_ctx = mod.DEFAULT_CONTEXT_TOKENS
        # Coherent pair property: EITHER both belong to writer B
        # (probe-success advanced the pair) OR both remain at the
        # pre-test hint state (probe-None left the pair alone). The
        # buggy shape produces (MODEL_A, V) — MODEL_A never has an
        # associated CTX in this test, and V belongs to MODEL_B.
        b_pair = (final_model == MODEL_B and final_ctx == V)
        pre_pair = (final_model == HINT_MODEL and final_ctx == HINT_CTX)
        self.assertTrue(
            b_pair or pre_pair,
            f"DA-round6 residual race: pair is not coherent — "
            f"(DEFAULT_MODEL={final_model!r}, "
            f"DEFAULT_CONTEXT_TOKENS={final_ctx!r}). "
            f"Expected either ({MODEL_B!r}, {V!r}) or "
            f"({HINT_MODEL!r}, {HINT_CTX!r}). Cross-model pair "
            f"({MODEL_A!r}, {V!r}) is the exact bug shape: writer "
            f"A wrote MODEL under strict-None while writer B's "
            f"CTX stayed. Fix: skip BOTH writes on strict-None.")

    def test_probe_ollama_reports_reachability_separately_when_source_is_hint(self):
        """DA-round4 (E): `probe_ollama()` compared `DEFAULT_MODEL in models`
        without considering that `DEFAULT_MODEL` may be
        `_DEFAULT_MODEL_HINT` (import fell through because the primary was
        down at import). A reachable host that never served the hint model
        was reported as "reachable but <hint> missing" — a false negative
        for reachability. Fix: when `_DEFAULT_MODEL_SOURCE == 'hint'` at
        probe time, report reachability as True and NOTE the hint-source
        rather than compare an import-time fallback to per-host truth.
        """
        # Import with primary down so the hint fires.
        path = os.path.join(self._tmp.name, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"url": "http://test.invalid:11434/api/chat"}, fh)
        os.environ["OLLAMA_OFFLOAD_CONFIG"] = path
        import urllib.request
        discovery_dead = {
            "/api/version": VERSION,
            "/api/ps": urllib.error.URLError("primary down at import"),
            "/api/tags": urllib.error.URLError("primary down at import"),
            "/api/show": SHOW,
        }
        urllib.request.urlopen = _routed_urlopen(discovery_dead)
        import ollama_offload
        mod = importlib.reload(ollama_offload)
        self.assertEqual(mod._DEFAULT_MODEL_SOURCE, "hint",
                         "precondition: import hit hint fallback")
        # Swap to a reachable host whose /api/tags names a DIFFERENT
        # model — the hint is NOT in that list. Old code returns
        # (False, "reachable but <hint> missing"); new code must return
        # (True, "...NOT compared...").
        mod.urllib.request.urlopen = _routed_urlopen({
            "/api/tags": {"models": [{"name": "other-model:test"}]},
        })
        ok, msg = mod.probe_ollama()
        self.assertTrue(
            ok,
            "reachable host must report reachable=True when the module "
            "DEFAULT_MODEL is a hint (import-time fallback), not compared "
            "to per-host truth")
        self.assertIn("hint", msg.lower(),
                      "message must name the hint-source so an operator can "
                      "act on it")
        self.assertNotIn(
            "missing", msg.lower(),
            "message must NOT report the hint as 'missing' — reachability "
            "and hint-source are distinct facts")


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
