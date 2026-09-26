#!/usr/bin/env python3
"""ollama_offload.py — a resilient client for offloading work to a self-hosted Ollama.

The offload pattern this supports: a local model does the wide, cheap, mechanical pass
over many items, and an expensive model (or a human) keeps the judgment calls. That only
works if the cheap pass is reliable, so this module is the single place that speaks to the
Ollama API and the single place that enforces the output contract.

Contract:
- Input:  a task prompt, plus a JSON schema describing the expected output.
- Output: a dict validated against that schema, or an OllamaCallError.
- Model:  discovered at runtime from the host, with a configured hint as fallback.
- Auth:   none. Ollama's default posture; keep it on a trusted network.

What it handles that a bare HTTP call does not: runtime model discovery, context-window
discovery, per-host cooldown, exponential backoff, cross-host failover, and -- the one
most implementations get wrong -- separating transport faults from model faults. A 200
response carrying unparseable content is the MODEL failing, not the endpoint. Putting a
healthy endpoint into cooldown for that punishes every other consumer sharing it.

Treat a local model's output as a probe rather than a verdict. Schema validation proves
the shape is right; it says nothing about whether the answer is.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# UTF-8 self-defense. On Windows the default stdout encoding is cp1252, so echoing a
# prompt containing any non-Latin-1 character raises UnicodeEncodeError mid-call.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Config: a JSON file next to this module, with environment overrides taking precedence.
# Models, URLs and thresholds all move over time, so nothing here is compiled in.
#
# OLLAMA_OFFLOAD_CONFIG relocates the file. That matters when this module is vendored
# into another tree read-only: without it the config must live inside the vendored copy,
# so every update overwrites local settings and a caller's real hosts cannot be kept
# outside the dependency.
_CONFIG_PATH = Path(
    os.environ.get("OLLAMA_OFFLOAD_CONFIG")
    or (Path(__file__).parent / "ollama_offload_config.json")
)
_CONFIG_SCHEMA_PATH = Path(__file__).parent / "ollama_offload_config.schema.json"

# Used only when runtime discovery in `_discover_model` cannot reach ANY
# endpoint (both /api/ps and /api/tags failed) AND neither the resolved host
# entry nor the config's top-level `model` names one. When either the host
# entry or the top-level config OMITS `model`, `_resolve_active_host` returns
# None for the model field so `_call_ollama_once` falls through to
# `DEFAULT_MODEL` (the runtime-discovered value). See :163/:189/:230.
_DEFAULT_MODEL_HINT = "qwen3.6:35b"


# ---------------------------------------------------------------------------
# Multi-host resolution. The config carries a `hosts.<name>` map and a
# `default_host`. OLLAMA_OFFLOAD_HOST selects an entry for this invocation,
# otherwise `default_host` applies, otherwise "primary".
#
# Top-level `url` / `model` in the config act as a fallback for a host entry
# that is absent or missing fields, which keeps a single-host config as short
# as it should be: set two keys and never write a `hosts` block at all.
# ---------------------------------------------------------------------------

class OllamaConfigError(RuntimeError):
    """Startup-time config error — an OLLAMA_OFFLOAD_HOST env explicitly names a host
    not present in ``ollama_offload_config.json``:``hosts``. Distinct from
    ``OllamaCallError`` (call-time) so callers can differentiate a
    misconfiguration (fix env / config) from a call failure (retry / cooldown).
    Raised only when the env var is EXPLICITLY set to an unknown name; an
    implicit fallback via ``default_host`` still legacy-tolerates a missing
    entry via top-level ``url``/``model`` for backward compat with pre-2026-
    07-02 configs (see the README).

    ALSO raised at load time by :func:`_load_config`
    when the config file fails JSON-Schema validation against
    ``ollama_offload_config.schema.json``. Catches the silent-coerce drift
    class (typo in ``strategy``, wrong type on ``hosts.<name>.weight``, etc.)
    before the caller sees a mysterious silent fallback."""


def _validate_config_schema(cfg: dict) -> None:
    """Validate the config dict against ``ollama_offload_config.schema.json``.

    Fail-loud on shape drift via OllamaConfigError.
    Fail-open if the schema file is missing OR the ``jsonschema`` package
    isn't importable (never break loading on a validation-infrastructure gap).
    """
    if not _CONFIG_SCHEMA_PATH.exists():
        return
    try:
        import jsonschema  # optional dependency
    except Exception:
        return
    try:
        schema = json.loads(_CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    try:
        jsonschema.validate(instance=cfg, schema=schema)
    except jsonschema.ValidationError as e:
        # Compact the path for the operator: hosts -> primary -> weight
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise OllamaConfigError(
            f"ollama_offload_config.json failed schema validation at "
            f"path={path!r}: {e.message}. Fix: correct the config against "
            f"ollama_offload_config.schema.json."
        ) from e


def _load_config() -> dict:
    """Load ollama_offload_config.json; return empty dict on any failure (defaults will apply).

    after a successful load, validate against
    ``ollama_offload_config.schema.json`` — a schema mismatch raises
    :class:`OllamaConfigError` (fail-loud on drift, distinct from an
    absent/malformed file which stays fail-open with defaults)."""
    try:
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    _validate_config_schema(cfg)
    return cfg


_CFG = _load_config()


def _resolve_active_host() -> dict:
    """Resolve the active host entry. Mirrors ``_ResolveActiveHost`` in the
    PowerShell twin; the two must agree, since they share a cooldown sidecar.

    Precedence:

    1. ``OLLAMA_OFFLOAD_URL`` → raw chat URL, bypassing the ``hosts`` map. The
       name is recorded as the ``<env-url>`` sentinel so an audit trail shows
       that routing came from the environment rather than from config.
    2. ``OLLAMA_OFFLOAD_HOST`` → picks ``hosts.<name>``. If it names a host
       absent from a non-empty ``hosts`` map, raises :class:`OllamaConfigError`
       rather than falling back — a typo that silently routes to the default
       host is indistinguishable from success.
    3. ``config.default_host`` → picks ``hosts.<default_host>``.
    4. Top-level ``config.url`` / ``config.model``, when the resolved entry is
       absent or missing a field. This is what lets a single-host setup skip
       the ``hosts`` block entirely.
    5. Last resort, the literal ``http://localhost:11434/api/chat``.
    """
    # Rule 1 + 2: raw-URL env short-circuits the hosts map entirely (PowerShell parity —
    # psm1 `_ResolveActiveHost` L126-131). Name sentinel matches PS so cross-
    # twin audit rows for the same env emit the same string
    # host_name — audit reproducibility contract).
    raw_url = os.environ.get("OLLAMA_OFFLOAD_URL")
    if raw_url:
        return {
            "name": "<env-url>",
            "url": raw_url,
            # None (config OMITS `model`) preserved through so
            # `_call_ollama_once` falls through to DEFAULT_MODEL (the runtime-
            # discovered value from `_discover_model`). Prior shape
            # `_CFG.get("model") or _DEFAULT_MODEL_HINT` shadowed the
            # discovered model on every env-URL call — the wire body kept
            # whatever the config pinned, contradicting the module's own
            # contract at :12 and README:46-47 (and dropping README:136's
            # documented OLLAMA_OFFLOAD_MODEL env override).
            "model": _CFG.get("model"),
            "context_tokens_fallback": _CFG.get("context_tokens_fallback") or 8192,
        }

    env_host = os.environ.get("OLLAMA_OFFLOAD_HOST")
    host_name = env_host or _CFG.get("default_host") or "primary"
    hosts_map = _CFG.get("hosts", {})
    if not isinstance(hosts_map, dict):
        hosts_map = {}
    entry = hosts_map.get(host_name)
    if env_host and entry is None and hosts_map:
        known = ", ".join(sorted(hosts_map.keys())) or "<none>"
        raise OllamaConfigError(
            f"OLLAMA_OFFLOAD_HOST={env_host!r} but no such host in "
            f"ollama_offload_config.json:hosts (known: [{known}]). "
            f"Fix: either add hosts.{env_host} to the config, unset "
            f"OLLAMA_OFFLOAD_HOST to use default_host={_CFG.get('default_host', 'primary')!r}, "
            f"or set OLLAMA_OFFLOAD_URL for a one-off raw-URL override."
        )
    if entry is None:
        entry = {}
    # Merge legacy top-level fields as fallback so a config that hasn't been
    # upgraded to the hosts-map still works (implicit-fallback path only).
    resolved = {
        "name": host_name,
        "url": entry.get("url") or _CFG.get("url") or "http://localhost:11434/api/chat",
        # Preserve None (both entry and config OMIT `model`) through so
        # `_call_ollama_once` falls through to DEFAULT_MODEL — the runtime-
        # discovered value from `_discover_model`. Prior shape shadowed the
        # discovered model with `_DEFAULT_MODEL_HINT` on every call; see
        # :163 for the same root cause and the module's own contract at :12.
        "model": entry.get("model") or _CFG.get("model"),
        "context_tokens_fallback": (
            entry.get("context_tokens_fallback")
            or _CFG.get("context_tokens_fallback")
            or 8192
        ),
    }
    return resolved


_ACTIVE_HOST = _resolve_active_host()


def _resolve_host_by_name(name: str) -> dict:
    """Resolve a specific host entry BY NAME (per-call host override path).

    Twin of the PS ``_ResolveHostByName``. Same class-of-error as
    :func:`_resolve_active_host` rule 3 — an unknown name raises
    :class:`OllamaConfigError` (typo trap symmetric with the env-driven path).
    Distinct from `_resolve_active_host` because the LATTER caches +
    honors ENV precedence; this one is a pure lookup + never caches.
    Used by :func:`call_ollama` when the caller passes ``host=<name>`` to
    route a single call to a specific configured host (drawer §Non-goals

    """
    hosts_map = _CFG.get("hosts", {})
    if not isinstance(hosts_map, dict):
        hosts_map = {}
    entry = hosts_map.get(name)
    if entry is None:
        known = ", ".join(sorted(hosts_map.keys())) or "<none>"
        raise OllamaConfigError(
            f"call_ollama(host={name!r}) but no such host in "
            f"ollama_offload_config.json:hosts (known: [{known}]). "
            f"Fix: add hosts.{name} to the config, or omit host= to use "
            f"the module-resolved active host "
            f"({_ACTIVE_HOST.get('name', 'unknown')!r})."
        )
    resolved = {
        "name": name,
        "url": entry.get("url") or _CFG.get("url") or "http://localhost:11434/api/chat",
        # Preserve None so `_call_ollama_once` falls to DEFAULT_MODEL. See :163.
        "model": entry.get("model") or _CFG.get("model"),
        "context_tokens_fallback": (
            entry.get("context_tokens_fallback")
            or _CFG.get("context_tokens_fallback")
            or 8192
        ),
    }
    return resolved

# Backward-compat name: `OLLAMA_OFFLOAD_URL` still exported as the module-level URL for
# any legacy caller that reads it directly. Now sourced from the resolved
# active host (so the raw-URL env short-circuit at `_resolve_active_host`
# rule 1/2 is honored — an earlier shape read env AGAIN here, which
# double-applied the env but also masked the drawer's precedence contract).
# New callers should use `_active_host_url()` / `_resolve_active_host()`.
OLLAMA_OFFLOAD_URL = _ACTIVE_HOST["url"]
DEFAULT_TIMEOUT_S = int(
    os.environ.get("OLLAMA_OFFLOAD_TIMEOUT_S")
    or os.environ.get("OLLAMA_OFFLOAD_TIMEOUT_S")
    or _CFG.get("timeout_s")
    or 300
)
_DEFAULT_TEMPERATURE = float(
    os.environ.get("OLLAMA_OFFLOAD_TEMPERATURE")
    or os.environ.get("OLLAMA_OFFLOAD_TEMPERATURE")
    or _CFG.get("temperature")
    or 0.1
)


def _resolve_default_think() -> bool | None:
    """Resolve the module-level default for Ollama /api/chat `think` field.

    Precedence: `OLLAMA_OFFLOAD_THINK` → `_CFG.get("think")` → None. The
    tri-state `None` means "omit the `think` field entirely", which keeps the
    emitted body identical to what a client that knows nothing about `think`
    would send. `1/true/yes/on` (case-insensitive) is truthy, anything else is
    falsy, and an explicit `None` in config stays None.
    """
    for src in (os.environ.get("OLLAMA_OFFLOAD_THINK"),):
        if src is None or src == "":
            continue
        return src.strip().lower() in ("1", "true", "yes", "on")
    cfg_val = _CFG.get("think")
    if cfg_val is None:
        return None
    if isinstance(cfg_val, bool):
        return cfg_val
    return str(cfg_val).strip().lower() in ("1", "true", "yes", "on")


_DEFAULT_THINK: bool | None = _resolve_default_think()


def _resolve_default_keep_alive() -> int | str | None:
    """Resolve the module-level default for Ollama /api/chat `keep_alive`.

    Precedence: `OLLAMA_OFFLOAD_KEEP_ALIVE` → `_CFG.get("keep_alive")` →
    ``-1`` (integer, resident). Set this explicitly on every call. Leaving it
    out lets the server apply its own ``"5m"`` default, which overrides any
    TTL you pinned elsewhere and evicts a warm model between calls — measured
    here as a 13.2 s cold-load stall that presented as a slow model rather
    than as an eviction.

    Ollama's ``keep_alive`` field accepts either an INTEGER (seconds,
    with ``-1`` meaning "keep forever" and ``0`` meaning "unload
    immediately") OR a duration STRING like ``"30m"`` / ``"1h"``. A
    string ``"-1"`` is REJECTED with HTTP 400 ``time: missing unit in
    duration "-1"`` because Ollama parses string values as Go
    ``time.Duration`` (which requires a unit suffix). The integer form
    is the canonical way to say "forever."

    The default is the integer ``-1``: this client owns the model-lifetime
    decision, and a caller wanting a bounded TTL passes an explicit
    ``keep_alive="30m"`` or ``keep_alive=1800``. An env override that parses
    as an integer is sent as one; anything else is treated as a duration
    string. An explicit ``None`` in config stays None, the same tri-state as
    ``think``, for callers who want the field omitted.
    """
    for src in (os.environ.get("OLLAMA_OFFLOAD_KEEP_ALIVE"),):
        if src is None or src == "":
            continue
        s = src.strip()
        try:
            return int(s)
        except ValueError:
            return s
    if "keep_alive" not in _CFG:
        return -1
    cfg_val = _CFG.get("keep_alive")
    if cfg_val is None:
        return None
    if isinstance(cfg_val, int):
        return cfg_val
    s = str(cfg_val).strip()
    try:
        return int(s)
    except ValueError:
        return s


_DEFAULT_KEEP_ALIVE: int | str | None = _resolve_default_keep_alive()


def _api_base() -> str:
    """Derive the Ollama root URL from OLLAMA_OFFLOAD_URL (strip /api/chat)."""
    return OLLAMA_OFFLOAD_URL.rsplit("/api/", 1)[0]


def active_host_name() -> str:
    """Public accessor: which host entry is the wrapper currently talking to?
    Useful for consumers that want to record the source in their output for
    audit / reproducibility."""
    return _ACTIVE_HOST["name"]


def _discover_model(*, base_url: str | None = None) -> str:
    """Learn the primary chat model DYNAMICALLY from the host at runtime.

    Strategy: /api/ps first (models currently loaded in RAM — the authoritative "what's
    actually serving right now"); if empty, /api/tags (all installed models). Filter out
    embedding models; prefer the largest by parameter count / size. OLLAMA_OFFLOAD_MODEL env
    override wins if set (for ops override or CI pinning). Fail-open to the last-known
    default if the API is unreachable — the actual call will error clearly later.

    ``base_url`` overrides the probe target — pass the resolved per-call base
    when a runtime caller has selected a non-active host (via ``call_ollama(host=…)``
    or a failover). Precedent: :641 ``ollama_alive(base_url=)``. Default of ``None``
    preserves import-time behaviour of probing the module-active host.
    """
    global _DEFAULT_MODEL_SOURCE
    override = os.environ.get("OLLAMA_OFFLOAD_MODEL")
    if override:
        _DEFAULT_MODEL_SOURCE = "env"
        return override

    def _pick_best(models: list[dict]) -> str | None:
        # models = [{"name": "qwen3:8b", "size": ...}, ...] from /api/tags or /api/ps
        chat = [m for m in models if "embedding" not in m.get("name", "").lower()]
        if not chat:
            return None
        # Prefer largest by 'size' if present (bytes on disk / in-RAM), else name-sort as fallback
        chat_with_size = [m for m in chat if isinstance(m.get("size"), (int, float))]
        if chat_with_size:
            return max(chat_with_size, key=lambda m: m["size"])["name"]
        return sorted(m["name"] for m in chat)[-1]  # lexicographic — later versions sort higher

    base = base_url if base_url is not None else _api_base()
    for endpoint in ("/api/ps", "/api/tags"):
        try:
            with urllib.request.urlopen(urllib.request.Request(base + endpoint), timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        models = payload.get("models") or []
        picked = _pick_best(models)
        if picked:
            _DEFAULT_MODEL_SOURCE = "ps" if endpoint == "/api/ps" else "tags"
            return picked

    # Everything failed — return a config-file value or a last-resort literal so the actual
    # call still has a name to send. It'll error explicitly if that name isn't installed.
    # Also mark the module-level source flag: consumers can force a rediscovery at first call
    # instead of shipping a stale hint to a REACHABLE per-call host that was never asked
    # (F5 in the 2026-09-26 DA-Fable review of 3987814).
    cfg_pin = _CFG.get("model")
    if cfg_pin:
        _DEFAULT_MODEL_SOURCE = "cfg"
        return cfg_pin
    _DEFAULT_MODEL_SOURCE = "hint"
    return _DEFAULT_MODEL_HINT


# Source of DEFAULT_MODEL, set by _discover_model. Values:
#   "env" — OLLAMA_OFFLOAD_MODEL was set at import
#   "ps"  — /api/ps returned a model on the module-active host
#   "tags"— /api/tags returned a model
#   "cfg" — top-level `_CFG["model"]` was set and discovery failed
#   "hint"— everything failed; the hardcoded _DEFAULT_MODEL_HINT was used
# The wire-body resolution below forces a fresh discovery against the per-call
# host whenever this is "hint", so a REACHABLE failover/override host does not
# receive the import-time hint's homelab-specific name.
_DEFAULT_MODEL_SOURCE: str = "unknown"


DEFAULT_MODEL = _discover_model()


def _discover_context_tokens(model: str, *, base_url: str | None = None) -> int:
    """Learn the model's LOADED context window DYNAMICALLY (the runtime num_ctx that
    Ollama is actually serving), not the architectural maximum.

    Ollama exposes TWO context numbers and they differ:
    - /api/ps `context_length` per running model = the RUNTIME context (operator's
      num_ctx choice via Modelfile / OLLAMA_CONTEXT env). This is what actually gets
      used per request.
    - /api/show model_info.<arch>.context_length = the ARCHITECTURAL max (native
      pretraining ceiling). Often much larger than what's loaded (e.g. 262144 vs 98304
      when VRAM pressure or KV-cache quantization forces a smaller load).

    We MUST use the runtime value or we will over-fill prompts. Probe pattern:
    a mismatch between architectural max and runtime is normal (operator chooses),
    but the RUNTIME value is what serves the request.

    ``base_url`` overrides the probe target for the per-call host — see
    :func:`_discover_model` for the same pattern. Without it a runtime probe
    for a failed-over or ``host=<name>``-overridden call would hit the
    module-active host, sending the wrong host's context to the right host.
    """
    override = os.environ.get("OLLAMA_OFFLOAD_CONTEXT_TOKENS")
    if override:
        try:
            return int(override)
        except ValueError:
            pass

    base = base_url if base_url is not None else _api_base()

    # PRIMARY: /api/ps (the LOADED runtime context — what actually gets served).
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/api/ps"), timeout=5) as resp:
            ps = json.loads(resp.read().decode("utf-8"))
        for m in ps.get("models") or []:
            if m.get("name") == model or m.get("model") == model:
                ctx = m.get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
        # Model not in /api/ps -> not loaded. Fall through; the first call will load it.
    except Exception:
        pass

    # FALLBACK: /api/show -> num_ctx from parameters (Modelfile-declared runtime context).
    try:
        req = urllib.request.Request(
            base + "/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return int(_CFG.get("context_tokens_fallback", 8192))

    params = info.get("parameters", "") or ""
    if isinstance(params, str):
        for line in params.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                try:
                    return int(parts[1])
                except ValueError:
                    pass

    # LAST RESORT: architectural context_length (may over-report the actual loaded limit,
    # but strictly better than a hardcoded assumption).
    model_info = info.get("model_info") or {}
    for key, val in model_info.items():
        if key.endswith(".context_length") and isinstance(val, int):
            return val

    return int(_CFG.get("context_tokens_fallback", 8192))


DEFAULT_CONTEXT_TOKENS = _discover_context_tokens(DEFAULT_MODEL)


def get_consumer_config(section: str) -> dict:
    """Consumers call this to fetch their config section from ollama_offload_config.json.
    Example: get_consumer_config('summarizer') -> {'context_usage_ratio': 0.4, ...}"""
    return _CFG.get(section) or {}


def context_bytes(ratio: float, *, chars_per_token: int = 4) -> int:
    """Derive an input byte budget from the DISCOVERED model context. `ratio` is the
    fraction of the model's context window to spend on this input (e.g. 0.3 = 30%).
    chars_per_token ~= 4 for qwen family. Consumers use this to size input caps so a
    bigger-context model automatically gets bigger caps (no hardcoded '98K')."""
    return int(DEFAULT_CONTEXT_TOKENS * ratio * chars_per_token)


class OllamaCallError(RuntimeError):
    """Any failure talking to the Ollama host / parsing/validating the response."""


class OllamaUnavailable(OllamaCallError):
    """Distinct-subclass signal that the host is down / rate-limited / in cooldown.
    Consumers can decide whether to fail-open (return UNCERTAIN, skip) or hard-error."""


# ---------------------------------------------------------------------------
# Resilience: ollama_alive() pre-flight, exponential backoff, persistent cooldown.
# The shape is a standard rate-limit bailout: pre-flight probe, retry with
# backoff, and on repeated failure write a cooldown-until timestamp to a
# sidecar file so later runs no-op cleanly instead of shredding the workload
# against a host that is already known to be down.
# ---------------------------------------------------------------------------

_COOLDOWN_PATH = Path(os.environ.get("TEMP", "/tmp")) / "ollama_offload_cooldown.until"
_COOLDOWN_ON_FAIL_S = int(os.environ.get("OLLAMA_OFFLOAD_COOLDOWN_S") or _CFG.get("cooldown_on_fail_s", 300))
_MAX_RETRIES = int(os.environ.get("OLLAMA_OFFLOAD_MAX_RETRIES") or _CFG.get("max_retries", 3))
_BASE_BACKOFF_S = float(os.environ.get("OLLAMA_OFFLOAD_BASE_BACKOFF_S") or _CFG.get("base_backoff_s", 1.0))

# Consecutive-fail counter at module scope (per-process); a fresh Python invocation
# always starts from 0 — the PERSISTENT signal is the cooldown file below.
_consecutive_fails = 0


def _in_cooldown() -> tuple[bool, float]:
    """Return (in_cooldown, seconds_until_lapse) for the GLOBAL cooldown
    sidecar. Retained for backward-compat: legacy consumers wrote to
    ``_COOLDOWN_PATH`` directly. Reads the sidecar; treats a malformed /
    stale file as no-cooldown (fail-open).

    new code path is :func:`_in_cooldown_for` (per-host).
    The global path is treated as an "all hosts benched" signal for
    backward-compat during transition."""
    try:
        until = float(_COOLDOWN_PATH.read_text().strip())
        remaining = until - time.time()
        return (remaining > 0, max(0.0, remaining))
    except Exception:
        return (False, 0.0)


def _set_cooldown(reason: str, duration_s: int = None) -> None:
    """Write the GLOBAL cooldown sidecar. Retained for backward-compat.

    new code arms per-host via :func:`_set_cooldown_for`.
    Direct callers of ``_set_cooldown()`` bench every host; keep for tests
    that legacy callers exercised the global path directly."""
    dur = int(duration_s if duration_s is not None else _COOLDOWN_ON_FAIL_S)
    until = time.time() + dur
    try:
        _COOLDOWN_PATH.write_text(str(until))
        print(f"ollama_offload: cooldown until {time.strftime('%H:%M:%S', time.localtime(until))} — {reason}",
              file=sys.stderr)
    except Exception:
        pass  # cooldown is a hint; failure to write it is not fatal


def _sanitize_host_for_path(name: str) -> str:
    """Return a filesystem-safe token for use in a per-host cooldown filename.

    Preserves alphanumeric + hyphen + underscore; any
    other byte becomes underscore. Prevents path traversal (``../``) and
    OS-illegal chars (``:``, ``/``, ``\\``). ``<env-url>`` from the raw-URL
    bypass becomes ``_env-url_`` — a stable dedicated bucket, never colliding
    with a real host name (angle brackets never appear in a real name).

    db-sme Batch-2 NTH#4 fold: an empty/whitespace-only name would sanitize
    to ``''`` giving a filename ``ollama_offload_cooldown..until`` (double-dot). Not
    exploitable (host names are never empty in current code) but a defensive
    gap — coerce to ``_unnamed_`` when sanitize collapses to empty."""
    sanitized = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name)
    return sanitized or "_unnamed_"


def _cooldown_path_for(host_name: str) -> Path:
    """Per-host cooldown sidecar path: ``$TEMP/ollama_offload_cooldown.<host>.until``.

    Twin of PS ``_CooldownPathFor``. Distinct filename
    per host so benching ``hosts.secondary`` doesn't bench ``hosts.primary`` —
    the fine-grained shape the drawer's §Non-goals item 2 required for
    cross-host failover as a prerequisite."""
    safe = _sanitize_host_for_path(host_name)
    return Path(os.environ.get("TEMP", "/tmp")) / f"ollama_offload_cooldown.{safe}.until"


def _in_cooldown_for(host_name: str) -> tuple[bool, float]:
    """Return ``(in_cooldown, seconds_until_lapse)`` for the given host.

    Backward-compat OR: reads both the per-host
    sidecar AND the GLOBAL sidecar (``_COOLDOWN_PATH``). Either being in
    the future counts — the global path is treated as "all hosts benched"
    (legacy semantics). The returned ``seconds_until_lapse`` is the
    LONGER of the two.
    """
    max_remaining = 0.0
    in_cd = False
    for path in (_cooldown_path_for(host_name), _COOLDOWN_PATH):
        try:
            until = float(path.read_text().strip())
            remaining = until - time.time()
            if remaining > 0:
                in_cd = True
                if remaining > max_remaining:
                    max_remaining = remaining
            else:
                # db-sme Batch-2 NTH#3 fold: unlink expired sidecar (mirror
                # of the legacy _CooldownRemaining cleanup). Per-host sidecars
                # otherwise accumulate one file per host name ever benched.
                try:
                    path.unlink()
                except Exception:
                    pass
        except Exception:
            continue
    return (in_cd, max_remaining)


def _set_cooldown_for(host_name: str, reason: str, duration_s: int | None = None) -> None:
    """Arm the per-host cooldown sidecar. 

    Only benches ``host_name`` — other hosts remain callable. Prints a
    stderr line naming the benched host so an operator grepping the log
    can see WHICH host tripped which cooldown."""
    dur = int(duration_s if duration_s is not None else _COOLDOWN_ON_FAIL_S)
    until = time.time() + dur
    try:
        _cooldown_path_for(host_name).write_text(str(until))
        print(
            f"ollama_offload: host={host_name} cooldown until "
            f"{time.strftime('%H:%M:%S', time.localtime(until))} — {reason}",
            file=sys.stderr,
        )
    except Exception:
        pass  # cooldown is a hint; failure to write it is not fatal


def ollama_alive(timeout_s: int = 4, *, base_url: str | None = None) -> bool:
    """Pre-flight probe: ``/api/version``. Returns True iff Ollama is reachable.

    ``base_url``: when set, probes THAT root
    instead of the module-resolved ``_api_base()``. Used by
    :func:`call_ollama` when the caller passes ``host=<name>`` to route a
    single call to a specific configured host; the pre-flight then targets
    THAT host, not the cached module active host.
    """
    root = base_url if base_url is not None else _api_base()
    try:
        with urllib.request.urlopen(urllib.request.Request(root + "/api/version"),
                                    timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _call_ollama_once(
    task_prompt: str,
    schema: dict | None = None,
    *,
    model: str | None = None,
    system: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    think: bool | None = None,
    keep_alive: str | None = None,
    host: str | None = None,
    # Per-call temperature override, for parity with the PowerShell twin
    # `-Temperature`. `None` means "not supplied" and falls through to
    # _DEFAULT_TEMPERATURE -- NOT 0.0, because a 0.0 default would be
    # indistinguishable from a caller deliberately pinning 0 and would silently
    # override the configured default on every call that omitted the argument.
    temperature: float | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = 5,
) -> dict | str:
    """Single-shot Ollama call — one host, no failover.

    the pre-existing ``call_ollama`` body was
    renamed here; the public ``call_ollama`` now orchestrates failover across
    a chain of hosts, delegating to this helper for each individual attempt.

    Resilience (pre-flight probe, backoff, cooldown;
    a standard rate-limit bailout): checks the per-host cooldown
    first, then pre-flights ``ollama_alive()`` on the resolved
    host, then retries with exponential backoff on transient failures.
    Exhausted retries arm the per-host cooldown + raise ``OllamaUnavailable``
    so subsequent invocations no-op cleanly instead of shredding a batch
    through cascading timeouts. The public ``call_ollama`` catches that
    ``OllamaUnavailable`` and tries the next failover host.
    """
    global _consecutive_fails

    # per-call host override. When set, resolve the
    # named host at CALL time (via _resolve_host_by_name) + use its URL for
    # both pre-flight + POST. Unknown name raises OllamaConfigError — typo-trap
    # symmetric with the env-driven rule 3 in _resolve_active_host. Consumers
    # get the resolved host echoed back in _meta.active_host for audit.
    # (Resolve BEFORE cooldown check so the check keys on the RESOLVED host —
    # A per-host cooldown means a benched sibling does not block us.)
    if host is not None:
        _call_host = _resolve_host_by_name(host)
    else:
        _call_host = _ACTIVE_HOST
    _call_url = _call_host["url"]
    _call_base = _call_url.rsplit("/api/", 1)[0]
    # Precedence (highest first, per README:136 + docstring at :55-60):
    #   1. explicit ``model=`` per-call arg — caller's explicit intent
    #   2. OLLAMA_OFFLOAD_MODEL env — ops override / CI pin
    #   3. per-host config pin `hosts.<name>.model`
    #   4. runtime discovery on the resolved per-call host (only when the
    #      override host has no pin AND is not the module-active host — the
    #      active host's model was already discovered into DEFAULT_MODEL at
    #      import). Passes ``base_url=_call_base`` so the probe hits the
    #      RIGHT host, not the module-active one.
    #   5. DEFAULT_MODEL — the import-time snapshot of the active host.
    # Pre-3987814 this collapsed to a single truthy check on the pin, which
    # made env-override dead whenever config pinned a model — the exact
    # violation of README:136 the fix commit claimed to close.
    _env_model_override = os.environ.get("OLLAMA_OFFLOAD_MODEL")
    if model is not None:
        _effective_model = model
    elif _env_model_override:
        _effective_model = _env_model_override
    elif _call_host.get("model"):
        _effective_model = _call_host["model"]
    elif _call_host is not _ACTIVE_HOST:
        # Per-call host override with no pin — probe THIS host, not the
        # module-active one. Without this, a `call_ollama(host="secondary")`
        # would send the ACTIVE host's discovered model to the secondary,
        # exactly the failure the 3987814 follow-up (3) named.
        _effective_model = _discover_model(base_url=_call_base)
    elif _DEFAULT_MODEL_SOURCE == "hint":
        # F5: import-time discovery fell back to the hardcoded hint (primary
        # host was DOWN at import). Do NOT ship that hint to a REACHABLE host
        # that was never asked — a homelab-specific model name (`qwen3.6:35b`)
        # sent to a host that doesn't serve it becomes a 4xx, three retries,
        # and a cooldown that BENCHES A HEALTHY HOST on a model fault —
        # inverting the library's central fault-separation claim (:493-499).
        # Retry discovery against the per-call base first; if that ALSO
        # returns the hint, raise a clear config error rather than send a
        # probably-wrong name.
        _effective_model = _discover_model(base_url=_call_base)
        if _DEFAULT_MODEL_SOURCE == "hint":
            raise OllamaCallError(
                f"model discovery fell back to _DEFAULT_MODEL_HINT for host="
                f"{_call_host['name']} at {_call_base}. Set OLLAMA_OFFLOAD_MODEL, "
                f"add `model` to the host entry, or ensure /api/ps + /api/tags "
                f"answer with an installed model before calling."
            )
    else:
        _effective_model = DEFAULT_MODEL
    # Parity: this twin never sent num_ctx; the psm1 twin always did. Resolved
    # here beside the model because context length is a property OF the model -- a per-call
    # model override must not silently keep the default model's context window.
    # DEFAULT_CONTEXT_TOKENS is discovered once at import for DEFAULT_MODEL, so the common
    # path costs nothing; only a genuine model override pays for a discovery probe.
    # The identity guard on ``_call_host is _ACTIVE_HOST`` matters: without it,
    # a same-model fleet (host A + host B both serving qwen3.6:35b at different
    # num_ctx) reuses the module-active host's context on the wire to the
    # override host — a silent over-fill exactly of the kind :424 warns against.
    _effective_context_tokens = (
        DEFAULT_CONTEXT_TOKENS
        if (_effective_model == DEFAULT_MODEL and _call_host is _ACTIVE_HOST)
        else _discover_context_tokens(_effective_model, base_url=_call_base)
    )

    # persistent per-host cooldown check. Reads the
    # per-host sidecar OR the legacy GLOBAL sidecar (backward-compat "all
    # benched" signal). Skips until whichever is furthest-in-the-future lapses.
    in_cd, remaining = _in_cooldown_for(_call_host["name"])
    if in_cd:
        raise OllamaUnavailable(
            f"Ollama host={_call_host['name']} in cooldown for {remaining:.0f}s "
            f"more (touch/rm {_cooldown_path_for(_call_host['name'])} to clear "
            f"the per-host sidecar; {_COOLDOWN_PATH} clears the legacy global)"
        )

    # Pre-flight the RESOLVED host's base URL, not the module-cached one.
    # A per-call override to a dead ad-hoc host would otherwise pre-flight
    # the module host + succeed, then blow up in the retry loop.
    if not ollama_alive(base_url=_call_base):
        _set_cooldown_for(
            _call_host["name"],
            f"ollama_alive pre-flight failed for host={_call_host['name']}",
        )
        raise OllamaUnavailable(f"Ollama unreachable at {_call_base}")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": task_prompt})

    body = {
        "model": _effective_model,
        "messages": messages,
        "stream": False,
        # Parity: num_ctx joins temperature here. The psm1 twin sent num_ctx
        # ONLY and this twin sent temperature ONLY, so the same logical call produced a
        # different wire body per language -- and since the model's own default
        # temperature is 1, PowerShell callers were sampling at 1.0 while Python ran at
        # 0.1. Both keys now travel together in both twins.
        "options": {
            "temperature": (_DEFAULT_TEMPERATURE if temperature is None
                            else float(temperature)),
            "num_ctx": _effective_context_tokens,
        },
    }
    if schema is not None:
        body["format"] = schema

    # `think` is a TOP-LEVEL field on Ollama /api/chat (sibling of stream /
    # format / options — NOT nested in options; prior in-options placement
    # was a proven no-op). Guarded by `is not None` so the default None case
    # sends no field at all and the wire payload matches pre-`think` behaviour
    # for every existing caller.
    _effective_think = think if think is not None else _DEFAULT_THINK
    if _effective_think is not None:
        body["think"] = _effective_think

    # `keep_alive` is a TOP-LEVEL field on Ollama /api/chat controlling how
    # long the model stays warm after the call. Policy:
    # always set keep_alive explicitly on every direct call
    # MUST set it — absence lets Ollama apply its per-call "5m" default,
    # silently overriding operator TTL pins and evicting warm models between
    # calls. The SSOT default (`_DEFAULT_KEEP_ALIVE`) is int `-1` (forever)
    # so the wrapper OWNS the model-lifetime contract; explicit callers may
    # pass a bounded duration string like "30m" OR an int seconds count.
    # NOTE: string "-1" is REJECTED by Ollama (HTTP 400, missing time unit) —
    # the "forever" sentinel is int-only; see `_resolve_default_keep_alive`
    # docstring for the wire-shape rules. Tri-state None matches `think`:
    # an explicit config None sends no field, byte-for-byte compat with
    # pre-`keep_alive` behavior for callers that opt back out.
    _effective_keep_alive = keep_alive if keep_alive is not None else _DEFAULT_KEEP_ALIVE
    if _effective_keep_alive is not None:
        body["keep_alive"] = _effective_keep_alive

    # `logprobs` / `top_logprobs` are TOP-LEVEL fields on Ollama /api/chat, siblings of
    # think / keep_alive / format. When enabled the response carries a top-level `logprobs`
    # list of {token, logprob, bytes, top_logprobs:[{token, logprob, bytes}]}.
    #
    # WHY: a model asked for a verdict emits one token, and that token throws away how close
    # the call was. A model answering "1" may hold 0.55/0.45 internally, and the emitted token
    # renders that indistinguishable from 0.99/0.01. Reading the score token's logprobs
    # recovers the margin — this is G-Eval's probability-weighted scoring, and it turns a
    # binary flag back into a graded signal that downstream metrics can actually use.
    #
    # Default OFF and guarded by `if`, matching the tri-state convention `think` and
    # `keep_alive` already use here: an untouched caller sends a byte-identical wire payload.
    # Not free — the response grows with the token count, so this is opt-in per call rather
    # than a module default.
    if logprobs:
        body["logprobs"] = True
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs

    req = urllib.request.Request(
        _call_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    last_err: Exception | None = None
    payload = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            _consecutive_fails = 0  # success clears the streak
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last_err = e
            _consecutive_fails += 1
            if attempt < _MAX_RETRIES:
                delay = _BASE_BACKOFF_S * (2 ** (attempt - 1))
                print(f"ollama_offload: attempt {attempt}/{_MAX_RETRIES} failed ({e}); backoff {delay:.1f}s",
                      file=sys.stderr)
                time.sleep(delay)
            continue
    if payload is None:
        # All retries exhausted — arm the PER-HOST cooldown so
        # subsequent runs targeting THIS host no-op cleanly, while calls
        # to other hosts remain unblocked. Backward-compat: consumers still
        # blocking on the global sidecar keep working via _in_cooldown_for's
        # OR-read.
        _set_cooldown_for(
            _call_host["name"],
            f"{_MAX_RETRIES} consecutive failures ({last_err})",
        )
        raise OllamaUnavailable(
            f"Ollama host={_call_host['name']} call failed {_MAX_RETRIES}× "
            f"(last: {last_err}); per-host cooldown engaged"
        )

    content = payload.get("message", {}).get("content", "")
    elapsed = time.monotonic() - started

    if schema is None:
        return content

    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        raise OllamaCallError(
            f"Ollama content wasn't valid JSON (elapsed {elapsed:.1f}s): {e}\nRaw: {content[:400]}"
        ) from e

    # Minimal schema validation: check required top-level keys are present. Deep validation
    # is the consumer's job (each has its own domain checks); we just ensure the model
    # produced the top-level shape.
    for key in schema.get("required", []):
        if key not in obj:
            raise OllamaCallError(f"Ollama response missing required key '{key}': keys={list(obj)}")

    obj["_meta"] = {
        "model": _effective_model,
        "active_host": _call_host["name"],
        "elapsed_s": round(elapsed, 2),
        "eval_tokens": payload.get("eval_count"),
        "prompt_tokens": payload.get("prompt_eval_count"),
    }
    # Token-level probabilities ride in `_meta` rather than at the top level, so a
    # caller that did not ask for them sees a byte-identical dict and no schema validator
    # trips on an unexpected key. Present ONLY when the model actually returned them: absence
    # therefore means "not requested, or this Ollama build does not support it" — never an
    # empty list masquerading as a real-but-uninformative distribution.
    if logprobs and payload.get("logprobs"):
        obj["_meta"]["logprobs"] = payload["logprobs"]
    return obj


# Rotation counter. Process-scope Python int (arbitrary precision,
# no overflow). the counter is persisted to a shared
# sidecar `$TEMP/ollama_offload_rotation.next` so mixed Python+PS callers see a
# SYNCHRONIZED cursor rather than per-language drift. Fail-open on sidecar
# read/write errors (module counter is the fallback).
_ROTATION_COUNTER = 0
_ROTATION_SIDECAR = Path(os.environ.get("TEMP", "/tmp")) / "ollama_offload_rotation.next"
_VALID_STRATEGIES = ("primary", "round_robin", "weighted")


def _load_rotation_counter() -> int | None:
    """Read the shared rotation sidecar. Returns the persisted int or None
    when the file is absent / malformed / unreadable (fail-open)."""
    try:
        return int(_ROTATION_SIDECAR.read_text().strip())
    except Exception:
        return None


def _persist_rotation_counter(value: int) -> None:
    """Write the current counter to the shared sidecar. Fail-silent on IO
    error — the counter is a hint; the module fallback keeps working."""
    try:
        _ROTATION_SIDECAR.write_text(str(int(value)))
    except Exception:
        pass
_BAD_STRATEGY_SEEN: set[str] = set()


def _warn_once_bad_strategy(value: object) -> None:
    """Emit a one-shot stderr warning per unknown strategy value seen this
    process. hipaa-sme Batch-3 NTH-1 fold — invalid strategy strings coerce
    to 'primary' silently, hiding the root cause when an operator's
    rotation config isn't rotating."""
    key = repr(value)
    if key in _BAD_STRATEGY_SEEN:
        return
    _BAD_STRATEGY_SEEN.add(key)
    print(
        f"ollama_offload: unknown strategy {value!r}, defaulting to 'primary' "
        f"(valid: {_VALID_STRATEGIES})",
        file=sys.stderr,
    )


def _build_rotation_pool(strategy: str) -> list[str]:
    """Return the ordered rotation pool for a strategy. 

    ``round_robin`` yields each configured host name once (sorted for
    deterministic ordering across processes). ``weighted`` yields each
    host ``entry.weight`` times (default 1). ``primary`` (default) returns
    an empty list — no rotation happens, module-resolved active host is used.
    """
    hosts_map = _CFG.get("hosts") or {}
    if not isinstance(hosts_map, dict):
        hosts_map = {}
    names = sorted(hosts_map.keys())
    if strategy == "round_robin":
        return names
    if strategy == "weighted":
        pool: list[str] = []
        for name in names:
            entry = hosts_map[name] or {}
            weight = entry.get("weight") if isinstance(entry, dict) else None
            try:
                w = max(1, int(weight) if weight is not None else 1)
            except (TypeError, ValueError):
                w = 1
            pool.extend([name] * w)
        return pool
    return []  # 'primary' or unrecognized -> no rotation


def _resolve_consumer_host(consumer: str) -> str | None:
    """Return the affinity host name for ``consumer`` if configured.

    Reads ``_CFG[consumer].host``. Returns None when
    the section is absent, is not a dict, or has no ``host`` field. Absent
    consumer sections are LEGAL (consumers may opt in without a host); an
    UNKNOWN host name is caught by ``_resolve_host_by_name`` downstream and
    surfaced as ``OllamaConfigError``.
    """
    section = _CFG.get(consumer)
    if not isinstance(section, dict):
        return None
    host = section.get("host")
    if not isinstance(host, str) or not host.strip():
        return None
    return host


def _next_host_by_strategy(strategy: str) -> str | None:
    """Advance the module rotation counter modulo the pool; skip cooldown'd.

    ``primary`` returns None (caller falls back to
    default resolution). Empty pool returns None. When every candidate is
    cooldown'd, returns pool[0] anyway so the failover chain / orchestrator
    can decide (the ensuing call will re-check cooldown + raise cleanly).
    """
    global _ROTATION_COUNTER
    pool = _build_rotation_pool(strategy)
    if not pool:
        return None
    # load persisted counter (cross-language sync) so a mixed
    # Py+PS caller sees a SHARED cursor. Fail-open on read errors.
    persisted = _load_rotation_counter()
    if persisted is not None:
        _ROTATION_COUNTER = persisted
    n = len(pool)
    for _ in range(n):
        idx = _ROTATION_COUNTER % n
        _ROTATION_COUNTER += 1
        _persist_rotation_counter(_ROTATION_COUNTER)
        candidate = pool[idx]
        in_cd, _ = _in_cooldown_for(candidate)
        if not in_cd:
            return candidate
    return pool[0]


def call_ollama(
    task_prompt: str,
    schema: dict | None = None,
    *,
    model: str | None = None,
    system: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    think: bool | None = None,
    keep_alive: str | None = None,
    host: str | None = None,
    failover_hosts: list[str] | None = None,
    strategy: str | None = None,
    consumer: str | None = None,
    # Twin parity with psm1 -Temperature. None = use the configured default.
    temperature: float | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = 5,
) -> dict | str:
    """Call Ollama with optional cross-host failover.

    When ``failover_hosts`` (or ``config.failover_hosts``) is set, this
    orchestrator builds an ordered chain: [primary, *failover_hosts] with
    duplicates removed, then delegates to :func:`_call_ollama_once` for each.
    A :class:`OllamaUnavailable` from one host causes the orchestrator to try
    the next; a :class:`OllamaConfigError` (misconfig / typo trap) bubbles
    immediately — those cannot be recovered by trying another host.

    Args:
        host: pin the PRIMARY host for this call. When None, the
            module-resolved active host is used. Wins over ``consumer`` and
            ``strategy`` — explicit user intent trumps rotation.
        failover_hosts: ordered list of host names to try if the primary
            fails. When None, sources from ``config.failover_hosts`` (empty
            list means "no failover" = legacy behaviour).
        strategy: rotation strategy for primary-host selection.
            One of ``'primary'`` (default; legacy behaviour), ``'round_robin'``
            (each configured host in turn), or ``'weighted'`` (each host
            repeated ``hosts.<name>.weight`` times, default 1). Per-call value
            wins over ``config.strategy``. Only fires when ``host`` and
            ``consumer`` do not already pin one. Rotation counter is
            process-scope — see the module docstring for cross-twin drift.
        consumer: name of a consumer section (e.g. ``'summarizer'``,
            ``'classifier'``) — per-consumer host affinity. When
            the section carries a ``host`` field, that value becomes the
            primary. Wins over ``strategy`` but loses to explicit ``host=``.
            A consumer-pinned host still respects cooldown; pair with
            ``failover_hosts`` for resilience.

    Returns:
        Same shape as :func:`_call_ollama_once`. When failover triggered
        (a non-primary host served), the returned dict (schema mode) grows
        ``_meta.failover_chain`` listing the hosts tried in order —
        the last entry is the one that served.

    Raises:
        OllamaConfigError: primary or any failover host name is unknown.
        OllamaUnavailable: every host in the chain failed.
    """
    # per-consumer host affinity. When ``consumer`` is
    # named and its config section carries a ``host`` field, treat it as the
    # primary UNLESS the caller pinned host= (explicit intent wins). This
    # runs BEFORE strategy resolution so a consumer-pinned host is not
    # overwritten by round-robin/weighted rotation.
    if host is None and consumer is not None:
        _consumer_host = _resolve_consumer_host(consumer)
        if _consumer_host is not None:
            host = _consumer_host

    # strategy resolution — round-robin / weighted
    # rotate the PRIMARY host across the configured `hosts` map. Per-call
    # strategy= kwarg wins; falls back to config.strategy; defaults to
    # 'primary' (legacy behavior). Only fires when the caller has
    # NOT pinned a specific host= (which the user's per-call intent must
    # win over any rotation).
    _strategy = strategy if strategy is not None else (_CFG.get("strategy") or "primary")
    if _strategy not in _VALID_STRATEGIES:
        # hipaa-sme Batch-3 NTH-1 fold: an invalid strategy string silently
        # coerces to 'primary' — surface once/process on stderr so operators
        # can see WHY a config-driven rotation isn't rotating.
        _warn_once_bad_strategy(_strategy)
        _strategy = "primary"
    if host is None and _strategy in ("round_robin", "weighted"):
        _picked = _next_host_by_strategy(_strategy)
        if _picked is not None:
            host = _picked

    # DA Batch-2 NTH#4 fold: coerce non-list config values to [] so a typo
    # like `"failover_hosts": "host-b"` (single-string) does NOT char-iterate
    # into ['s', 'a', 'v', ...]. Per-call kwarg is typed by the caller.
    _failover_cfg = failover_hosts if failover_hosts is not None else (
        _CFG.get("failover_hosts") or []
    )
    if not isinstance(_failover_cfg, list):
        _failover_cfg = []
    if not _failover_cfg:
        # No failover configured — behave as a single-host client: one host, one try.
        return _call_ollama_once(
            task_prompt, schema,
            model=model, system=system, timeout_s=timeout_s,
            think=think, keep_alive=keep_alive, host=host,
            temperature=temperature,
            logprobs=logprobs, top_logprobs=top_logprobs,
        )
    # db-sme Batch-2 NTH#2 fold: legacy global cooldown active → skip the
    # whole chain (every host is benched by the global OR-read semantic).
    # Surfacing "global cooldown active" is more honest than "N hosts exhausted".
    in_cd_global, remaining = _in_cooldown()
    if in_cd_global:
        raise OllamaUnavailable(
            f"Legacy global cooldown active ({remaining:.0f}s remaining); "
            f"touch/rm {_COOLDOWN_PATH} to clear (failover_hosts={_failover_cfg})"
        )
    # db-sme Batch-2 NTH#1 fold: raw-URL sentinel (`<env-url>`) is not a
    # config host name — passing it to _call_ollama_once as host=... would
    # raise OllamaConfigError (not in hosts map). When the module active host
    # is the sentinel, route the FIRST attempt WITHOUT host= so it uses the
    # cached _ACTIVE_HOST directly; failovers still resolve normally.
    _primary_active = host if host is not None else _ACTIVE_HOST["name"]
    _use_sentinel_route = (
        host is None and _primary_active == "<env-url>"
    )
    chain: list[str] = [_primary_active]
    for h in _failover_cfg:
        if h not in chain:
            chain.append(h)
    last_err: Exception | None = None
    for i, h in enumerate(chain):
        try:
            result = _call_ollama_once(
                task_prompt, schema,
                model=model, system=system, timeout_s=timeout_s,
                think=think, keep_alive=keep_alive, host=h,
                temperature=temperature,
                logprobs=logprobs, top_logprobs=top_logprobs,
            )
            if isinstance(result, dict) and i > 0:
                # Failover fired — record the actual chain traversed for
                # audit reproducibility (drawer §Non-goals item 1 shape).
                result.setdefault("_meta", {})["failover_chain"] = chain[: i + 1]
            return result
        except OllamaUnavailable as e:
            last_err = e
            continue
    raise OllamaUnavailable(
        f"All {len(chain)} hosts exhausted (chain={chain}); "
        f"last error: {last_err}"
    )


def ollama_healthcheck(*, timeout_s: int = 4) -> list[dict]:
    """Probe ``/api/tags`` on EVERY host in ``ollama_offload_config.json:hosts``.

    Consumers call this for a status-dashboard
    view of which alternates are reachable and how loaded they are; strictly
    informational — does NOT trip the cooldown sidecar, does NOT auto-route.
    Failover and preferred-alternate selection landed alongside it;
    healthcheck itself remains informational and does not auto-route.

    Returns a list of dicts sorted by host name for deterministic ordering:

        [
            {
                "name": <str>, "url": <chat url>, "reachable": <bool>,
                "latency_ms": <int>, "model_count": <int|None>,
                "error": <str|None>  # truncated to 200 chars when set
            },
            ...
        ]

    Twin of PS ``Get-OllamaHealthcheck``. Both hit the SAME endpoint (``/api/tags``)
    so cross-language cross-check is a simple equality on ``name`` +
    ``reachable``.
    """
    hosts_map = _CFG.get("hosts", {})
    if not isinstance(hosts_map, dict):
        hosts_map = {}
    results: list[dict] = []
    for name in sorted(hosts_map):
        entry = hosts_map[name] or {}
        chat_url = entry.get("url") or _CFG.get("url") or "http://localhost:11434/api/chat"
        base = chat_url.rsplit("/api/", 1)[0]
        tags_url = base + "/api/tags"
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(tags_url), timeout=timeout_s
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latency_ms = int((time.monotonic() - started) * 1000)
            models = data.get("models") or []
            results.append({
                "name": name,
                "url": chat_url,
                "reachable": True,
                "latency_ms": latency_ms,
                "model_count": len(models),
                "error": None,
            })
        except Exception as e:
            latency_ms = int((time.monotonic() - started) * 1000)
            results.append({
                "name": name,
                "url": chat_url,
                "reachable": False,
                "latency_ms": latency_ms,
                "model_count": None,
                "error": str(e)[:200],
            })
    return results


def probe_ollama() -> tuple[bool, str]:
    """Quick reachability probe: returns (ok, message). URL derived from OLLAMA_OFFLOAD_URL."""
    tags_url = OLLAMA_OFFLOAD_URL.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        with urllib.request.urlopen(urllib.request.Request(tags_url), timeout=5) as resp:
            models = [m["name"] for m in json.loads(resp.read()).get("models", [])]
        if DEFAULT_MODEL in models:
            return True, f"Ollama reachable at {tags_url}; {DEFAULT_MODEL} present ({len(models)} models)"
        return False, f"Ollama reachable but {DEFAULT_MODEL} missing; found: {models}"
    except Exception as e:
        return False, f"Ollama unreachable at {tags_url}: {e}"


if __name__ == "__main__":
    # CLI: probe reachability (no args) OR pipe a prompt on stdin (--prompt).
    if len(sys.argv) == 1:
        ok, msg = probe_ollama()
        print(msg)
        sys.exit(0 if ok else 2)
    if sys.argv[1] == "--prompt":
        prompt = sys.stdin.read()
        try:
            out = call_ollama(prompt)
        except OllamaCallError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(out)
