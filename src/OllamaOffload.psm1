#Requires -Version 7.0
<#
.SYNOPSIS
  OllamaOffload.psm1 — PowerShell 7 twin of ollama_offload.py.
  One call surface (Invoke-OllamaCall) for any PowerShell consumer.

.DESCRIPTION
  Five properties this module is responsible for:
    1. Nothing hardcoded  - model, context window, timeouts and limits come from config
                            or from the host at runtime
    2. Dynamic discovery  - /api/ps and /api/show once per process, cached module-scope
    3. Congestion control - exponential backoff, then a persistent per-host cooldown
    4. Structured output  - optional -Schema hashtable, sent as /api/chat's 'format' field
    5. No policy gating   - this module transports; the calling pipeline decides what is
                            allowed to be sent and what may be done with the answer

  It shares state with the Python twin so mixed-language callers never fight each other:
  the same config file, the same cooldown sentinel under $env:TEMP, and the same
  OLLAMA_OFFLOAD_* environment overrides. A cooldown armed by a Python caller is therefore
  honoured by a PowerShell one, which is the point -- a backed-off endpoint should stay
  backed off regardless of which twin noticed.

  Usage:
    Import-Module ./OllamaOffload.psm1 -Force
    $cfg   = Get-OllamaConsumerConfig -Name 'summarizer'
    $bytes = if ($cfg.max_input_bytes_override) { [int]$cfg.max_input_bytes_override }
             else { Get-OllamaContextBytes -Ratio ($cfg.context_usage_ratio ?? 0.40) }
    try {
      $out = Invoke-OllamaCall -TaskPrompt $prompt -Schema $schema
    } catch [OllamaUnavailable] {
      # Degrade gracefully: the host is down or in cooldown. Return an empty payload
      # rather than failing the whole batch.
    }

.NOTES
  PowerShell 7 only: ternary (?:), null-coalescing (??), class-based exceptions and
  Invoke-RestMethod -TimeoutSec.

  Host selection:
    - OLLAMA_OFFLOAD_HOST picks `hosts.<name>` from the config. A name that is NOT in the
      map raises OllamaConfigError rather than falling back, because a typo that silently
      routes back to the default host is indistinguishable from success.
    - OLLAMA_OFFLOAD_URL supplies a raw /api/chat URL, bypassing the map entirely, for a
      one-off host.
    - Get-OllamaActiveHost returns the resolved name, so a caller can record which host
      actually answered.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---- Custom exception types ------------------------------------------------
# Mirrors the Python exception hierarchy:
#   OllamaCallError  = base for all call-time failures the consumer can recover from
#   OllamaUnavailable = subclass — cheap fast-fail during cooldown or reachability.
# The subclass relationship is a CONTRACT: consumers catch [OllamaCallError] to sweep
# BOTH branches; a bare `System.Exception` parent (previous shape) leaked past that catch.
# Order matters — OllamaCallError MUST be defined before OllamaUnavailable.
class OllamaCallError : System.Exception {
    OllamaCallError([string]$Msg) : base($Msg) {}
}
class OllamaUnavailable : OllamaCallError {
    OllamaUnavailable([string]$Msg) : base($Msg) {}
}
# Startup-time config error — distinct from OllamaCallError so callers can
# differentiate a misconfiguration (fix env / config) from a call failure
# (retry / cooldown). Raised only when OLLAMA_OFFLOAD_HOST is EXPLICITLY set to a
# name absent from the `hosts` map; Python parity with OllamaConfigError.
class OllamaConfigError : System.Exception {
    OllamaConfigError([string]$Msg) : base($Msg) {}
}

# ---- Paths, module-scope cache ---------------------------------------------
# The module and its config live side by side.
# Used only when runtime discovery in `_DiscoverModelForHost` cannot reach ANY
# endpoint (both /api/ps and /api/tags failed) AND neither the resolved host
# entry nor the config's top-level `model` names one. When either the host
# entry or the top-level config OMITS `model`, the resolvers now return $null
# for the model field so `_InvokeOllamaCallOnce` falls through to per-host
# discovery (the runtime-discovered value). Python parity: _DEFAULT_MODEL_HINT.
$Script:DefaultModelHint   = 'qwen3.6:35b'
# F5: source of the value _Discover / _DiscoverModelForHost most recently
# returned. Values: 'env', 'ps', 'tags', 'cfg', 'hint', 'unknown'. The call
# path forces a fresh discovery against the per-call base_url when this is
# 'hint', so a REACHABLE host that was never asked does not receive the
# import-time hint's homelab-specific model name. Python parity:
# _DEFAULT_MODEL_SOURCE (module global).
$Script:DefaultModelSource = 'unknown'
# OLLAMA_OFFLOAD_CONFIG relocates the config file, for the case where this module is
# vendored into another tree read-only and the caller's real hosts must live outside
# the dependency. Python parity: _CONFIG_PATH reads the same variable.
$Script:ConfigPath   = if ($env:OLLAMA_OFFLOAD_CONFIG) { $env:OLLAMA_OFFLOAD_CONFIG }
                       else { Join-Path $PSScriptRoot 'ollama_offload_config.json' }
$Script:CooldownPath = Join-Path $env:TEMP 'ollama_offload_cooldown.until'   # SHARED with Python
$Script:Discovered   = $null     # cached { Model, ContextTokens, KeepAlive, BaseUrl }
$Script:CachedConfig = $null
$Script:ActiveHost   = $null     # cached { Name, Url } — resolved once per process

# ---- Config loader (SSOT: ollama_offload_config.json) -----------------------
function Get-OllamaConfig {
    [CmdletBinding()] param()
    if ($Script:CachedConfig) { return $Script:CachedConfig }
    if (-not (Test-Path $Script:ConfigPath)) {
        throw [OllamaCallError]::new("ollama_offload_config.json not found at $($Script:ConfigPath)")
    }
    $Script:CachedConfig = Get-Content -Raw $Script:ConfigPath | ConvertFrom-Json -AsHashtable
    return $Script:CachedConfig
}

function Get-OllamaConsumerConfig {
    [CmdletBinding()] param([Parameter(Mandatory)] [string]$Name)
    # Consumer
    # sections sit at the TOP LEVEL of ollama_offload_config.json (summarizer, classifier,
    # and any others you define). There is NO `consumers`
    # wrapper — a nested lookup silently returns {} for every consumer, which then
    # falls through to hardcoded defaults + defeats config tuning.
    $cfg = Get-OllamaConfig
    return ($cfg[$Name] ?? @{})
}

# ---- Env-override helpers --------------------------------------------------
function _EnvOr {
    param([string]$EnvName, $Default)
    $v = [Environment]::GetEnvironmentVariable($EnvName)
    if ($null -ne $v -and $v -ne '') { return $v }
    return $Default
}

function _ResolveDefaultTemperature {
    <#
    .SYNOPSIS
      Default sampling temperature, mirroring the Python twin's precedence exactly.
    .DESCRIPTION
      Closes a measured parity gap. The two twins were sending
      DISJOINT `options` payloads:

          Python : options = @{ temperature = 0.1 }     <- no num_ctx
          PSM1   : options = @{ num_ctx = <discovered> } <- no temperature

      Each sent what the other omitted, so the same logical call produced a different
      wire body depending on which twin a consumer happened to import. PowerShell
      callers were getting the SERVER default temperature while Python callers got
      0.1 -- silently, with nothing in either module hinting at the difference.

      Precedence is copied from Python `_DEFAULT_TEMPERATURE` and must stay in step:
        OLLAMA_OFFLOAD_TEMPERATURE -> config.temperature -> 0.1
    #>
    $v = _EnvOr 'OLLAMA_OFFLOAD_TEMPERATURE' $null
    if ($null -eq $v -or $v -eq '') {
        $cfg = Get-OllamaConfig
        if ($null -ne $cfg -and $null -ne $cfg.temperature) { $v = $cfg.temperature }
    }
    if ($null -eq $v -or $v -eq '') { return 0.1 }
    return [double]$v
}

function _ResolveActiveHost {
    # Mirrors the Python twin's _resolve_active_host.
    # Precedence:
    #   1. OLLAMA_OFFLOAD_URL env — raw chat URL, bypasses the `hosts` map
    #      (name recorded as '<env-url>' for audit).
    #   2. OLLAMA_OFFLOAD_HOST env — picks `hosts.<name>`. If EXPLICIT + name absent from
    #      the map, raise OllamaConfigError (typo trap).
    #   3. `$cfg.default_host` → `hosts.<default_host>.url`.
    #   4. Implicit fallback to legacy top-level `$cfg.url` (backward compat with
    #      pre-multi-host configs, applied only when env not explicit).
    #   5. Last-resort literal http://localhost:11434/api/chat.
    # Cached module-scope so multiple callers see the same resolved host.
    if ($Script:ActiveHost) { return $Script:ActiveHost }

    $rawUrlEnv = _EnvOr 'OLLAMA_OFFLOAD_URL'   $null
    if (-not $rawUrlEnv) { $rawUrlEnv = _EnvOr 'OLLAMA_OFFLOAD_URL' $null }
    if ($rawUrlEnv) {
        $Script:ActiveHost = @{ Name = '<env-url>'; Url = $rawUrlEnv }
        return $Script:ActiveHost
    }

    $cfg         = Get-OllamaConfig
    $envHost     = _EnvOr 'OLLAMA_OFFLOAD_HOST' $null
    $hostName    = $envHost ?? ($cfg['default_host'] ?? 'primary')
    $hostsMap    = $cfg['hosts'] ?? @{}
    $hostEntry   = $hostsMap[$hostName]

    if ($envHost -and (-not $hostEntry) -and $hostsMap.Count -gt 0) {
        $known = ($hostsMap.Keys | Sort-Object) -join ', '
        if (-not $known) { $known = '<none>' }
        throw [OllamaConfigError]::new(
            "OLLAMA_OFFLOAD_HOST='$envHost' but no such host in " +
            "ollama_offload_config.json:hosts (known: [$known]). Fix: either " +
            "add hosts.$envHost to the config, unset OLLAMA_OFFLOAD_HOST to use " +
            "default_host='$($cfg['default_host'])', or set OLLAMA_OFFLOAD_URL / " +
            "OLLAMA_OFFLOAD_URL for a one-off raw-URL override."
        )
    }
    if (-not $hostEntry) { $hostEntry = @{} }
    $configUrl = $hostEntry['url'] ?? $cfg['url'] ?? 'http://localhost:11434/api/chat'
    $Script:ActiveHost = @{ Name = $hostName; Url = $configUrl }
    return $Script:ActiveHost
}

function _ResolveHostByName {
    # Twin of Python `_resolve_host_by_name`.
    # Pure lookup — no env precedence, no cache. Unknown name raises OllamaConfigError
    # (typo trap symmetric with _ResolveActiveHost rule 3). Used by Invoke-OllamaCall
    # when -Host <name> is passed to route a single call to a specific configured host.
    param([Parameter(Mandatory)] [string]$Name)
    $cfg      = Get-OllamaConfig
    $hostsMap = $cfg['hosts'] ?? @{}
    $entry    = $hostsMap[$Name]
    if (-not $entry) {
        $known = ($hostsMap.Keys | Sort-Object) -join ', '
        if (-not $known) { $known = '<none>' }
        throw [OllamaConfigError]::new(
            "Invoke-OllamaCall -Host '$Name' but no such host in " +
            "ollama_offload_config.json:hosts (known: [$known]). Fix: add " +
            "hosts.$Name to the config, or omit -Host to use the " +
            "module-resolved active host ('$((_ResolveActiveHost).Name)')."
        )
    }
    return @{
        Name  = $Name
        Url   = $entry['url'] ?? $cfg['url'] ?? 'http://localhost:11434/api/chat'
        # Preserve $null when both entry and cfg OMIT `model` so
        # `_InvokeOllamaCallOnce` falls through to per-host runtime
        # discovery (the discovered value from `_DiscoverModelForHost`).
        # Prior shape `?? $script:DefaultModelHint` shadowed the discovered
        # model on every -HostName call and dropped OLLAMA_OFFLOAD_MODEL
        # precedence. Python parity: _resolve_host_by_name at :246 —
        # `entry.get("model") or _CFG.get("model")` (no hint fallback).
        Model = $entry['model'] ?? $cfg['model']
        ContextTokensFallback = $entry['context_tokens_fallback'] ?? $cfg['context_tokens_fallback'] ?? 8192
    }
}

function Get-OllamaActiveHost {
    <#
    .SYNOPSIS
      Return the currently-resolved active host name — PS twin of Python
      `active_host_name()` (ollama_offload.py L141). Public accessor so PS
      consumers can record which Ollama host served the response in their
      audit output (source-of-truth for `hosts.<name>` selection).
    .OUTPUTS
      [string] the resolved host name. '<env-url>' when OLLAMA_OFFLOAD_URL
      was used to bypass the `hosts` map with a raw chat URL.
    .NOTES
      Throws `OllamaConfigError` if `OLLAMA_OFFLOAD_HOST` names a host absent from
      `ollama_offload_config.json:hosts` (typo trap — see the README
      the README for the precedence
      contract). This is distinct from `OllamaCallError` — consumers using
      `catch [OllamaCallError]` for fail-open MUST add a separate
      `catch [OllamaConfigError]` clause to surface misconfig to the operator.
    #>
    [CmdletBinding()] param()
    return (_ResolveActiveHost).Name
}

function _GetBaseUrl {
    # LEGACY entry-point retained for internal callers (_Discover, Test-OllamaReachable,
    # Invoke-OllamaCall). Composes _ResolveActiveHost with OLLAMA_OFFLOAD_OLLAMA_BASE_URL BC
    # override + tail-strip so callers get the API root (append /api/ps, /api/show,
    # /api/tags, /api/chat).
    $active    = _ResolveActiveHost
    $configUrl = $active.Url
    $url       = _EnvOr 'OLLAMA_OFFLOAD_OLLAMA_BASE_URL' $configUrl
    # Strip trailing `/api/<endpoint>` so callers get the ROOT and can build endpoints.
    return ($url -replace '/api/[^/]+/?$', '')
}

# ---- P2 — Dynamic discovery (/api/show, cached per process) ----------------
$Script:DiscoverCallCount = 0
function _Discover {
    # Test-scope spy: T20 asserts $Script:DiscoverCallCount == 0 after an
    # Invoke-OllamaCall -HostName path, pinning that per-call HostName skips
    # module discovery (Batch-1 db-sme fold; MF-1 pre-fix bug).
    $Script:DiscoverCallCount++
    if ($Script:Discovered) { return $Script:Discovered }
    $cfg     = Get-OllamaConfig
    $baseUrl = _GetBaseUrl
    $hint    = $cfg['model']                                        # config's preferred model (hint)
    $fallbk  = [int]($cfg['context_tokens_fallback'] ?? 8192)

    # F5: OLLAMA_OFFLOAD_MODEL env override wins at discovery (parity with
    # Python `_discover_model`). Also stamps $Script:DefaultModelSource so
    # the per-call ladder in `_InvokeOllamaCallOnce` can tell where the
    # active-host model came from.
    $envOverride = _EnvOr 'OLLAMA_OFFLOAD_MODEL' $null
    $modelName   = $null
    if ($envOverride) {
        $modelName = $envOverride
        $Script:DefaultModelSource = 'env'
    }

    # If no hint, ask /api/ps for currently-loaded models and pick first.
    if (-not $modelName) {
        $modelName = $hint
        if ($modelName) { $Script:DefaultModelSource = 'cfg' }
    }
    try {
        if (-not $modelName) {
            $ps = Invoke-RestMethod -Method Get -Uri "$baseUrl/api/ps" -TimeoutSec 8
            if ($ps.models -and $ps.models.Count -gt 0) {
                $modelName = $ps.models[0].name
                $Script:DefaultModelSource = 'ps'
            }
        }
    } catch { }   # fall through — will error at /api/show below if truly unreachable

    if (-not $modelName) {
        # F5: nothing chose a model. Match Python — fall back to the hardcoded
        # hint + stamp the source so the per-call ladder can force a fresh
        # discovery against the resolved per-call base_url. Prior shape
        # raised OllamaUnavailable here, but that pre-empts F5's retry.
        $modelName = $Script:DefaultModelHint
        $Script:DefaultModelSource = 'hint'
    }

    try {
        $show = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/show" `
                    -ContentType 'application/json' `
                    -Body (@{ name = $modelName } | ConvertTo-Json) -TimeoutSec 8
    } catch {
        # /api/show failed — intentionally undersized fallback so a truncated payload
        # surfaces the discovery failure loudly (matches Python behavior).
        $Script:Discovered = @{ Model = $modelName; ContextTokens = $fallbk; KeepAlive = '5m'; BaseUrl = $baseUrl }
        return $Script:Discovered
    }

    # Bug fix: this twin could not complete ANY call against a reasoning model.
    #
    # `/api/show` returns `parameters` as a multi-line STRING, not an object:
    #     temperature   1
    #     top_k        20
    #     ...            (and no num_ctx at all for this model)
    # so `$show.parameters.num_ctx` is a property access ON A STRING. Under
    # `Set-StrictMode -Version Latest` that THROWS PropertyNotFoundException instead of
    # yielding $null, and the exception surfaced at the Invoke-OllamaCall frame, which made
    # it read like a call-path failure rather than a discovery one.
    #
    # The Python twin does the equivalent lookup with .get() and quietly returns None, so
    # the two languages disagreed on the SEMANTICS of a missing key, not on the logic. That
    # is the sharpest form of twin drift: identical code shape, divergent behaviour, and
    # only one side fails loudly.
    # ⭐ PRIMARY SOURCE = /api/ps, NOT model_info. and the bug it replaces
    # was mine from earlier the same day.
    #
    # Ollama exposes TWO context numbers and they are NOT interchangeable:
    #   /api/ps    models[].context_length = the LOADED RUNTIME context (what is actually
    #                                        being served right now)          -> 98304
    #   /api/show  model_info.<arch>.context_length = the ARCHITECTURAL MAX   -> 262144
    #
    # My earlier parse fix took this twin from "throws on every call" to "works but requests
    # the ARCHITECTURAL max", because the params text carries no num_ctx and the code fell
    # straight through to the model_info scan. Requesting 262144 against a model loaded at
    # 98304 forces Ollama to RELOAD 24 GB at 2.67x the KV cache, which spills to CPU — a
    # single call went from ~8.7s to >240s. Worse, the Python twin sends 98304, so the two
    # twins THRASHED the model: every alternation paid a full reload.
    #
    # Python's _discover_context_tokens already documented this ("/api/ps ... is what
    # actually gets served"). I used the source its docstring warns against. Same precedence
    # here now: /api/ps -> params num_ctx -> model_info (architectural, last resort) ->
    # config fallback.
    $ctxTokens = 0
    try {
        $ps = Invoke-RestMethod -Uri "$baseUrl/api/ps" -TimeoutSec 5
        if ($null -ne $ps -and $ps.PSObject.Properties.Name -contains 'models') {
            foreach ($m in $ps.models) {
                if ($m.name -eq $modelName -and $m.PSObject.Properties.Name -contains 'context_length') {
                    $ctxTokens = [int]$m.context_length; break
                }
            }
        }
    } catch { }   # not loaded / probe failed -> fall through to the declared sources below

    $paramText = if ($show.PSObject.Properties.Name -contains 'parameters') { $show.parameters } else { $null }
    if ($paramText -is [string] -and $paramText) {
        $m = [regex]::Match($paramText, '(?m)^\s*num_ctx\s+(\d+)\s*$')
        if ($m.Success) { $ctxTokens = [int]$m.Groups[1].Value }
    }
    elseif ($null -ne $paramText -and $paramText.PSObject.Properties.Name -contains 'num_ctx') {
        $ctxTokens = [int]$paramText.num_ctx      # object form, older Ollama vintages
    }
    if ($ctxTokens -le 0 -and $show.PSObject.Properties.Name -contains 'model_info') {
        # arch key varies; scan for any *.context_length
        foreach ($k in $show.model_info.PSObject.Properties.Name) {
            if ($k -like '*.context_length') { $ctxTokens = [int]$show.model_info.$k; break }
        }
    }
    if ($ctxTokens -le 0) { $ctxTokens = $fallbk }

    # Same String-vs-object trap as num_ctx above — keep_alive lives in the SAME
    # multi-line parameters text, so the object-style access would throw identically.
    $keepAlive = '5m'
    if ($paramText -is [string] -and $paramText) {
        $km = [regex]::Match($paramText, '(?m)^\s*keep_alive\s+(\S+)\s*$')
        if ($km.Success) { $keepAlive = $km.Groups[1].Value }
    }
    elseif ($null -ne $paramText -and $paramText.PSObject.Properties.Name -contains 'keep_alive') {
        $keepAlive = $paramText.keep_alive
    }

    $Script:Discovered = @{ Model = $modelName; ContextTokens = $ctxTokens; KeepAlive = $keepAlive; BaseUrl = $baseUrl }
    return $Script:Discovered
}

# ---- Per-host discovery helpers (F3: base_url threading) -------------------
# Twins of Python `_discover_model(base_url=)` + `_discover_context_tokens(base_url=)`.
# UNCACHED, one probe per call: caching would defeat per-host discovery when
# `-HostName` targets a host that is NOT the module-active one. The active-
# host path continues to use `_Discover` (cached module-scope) — these helpers
# only fire when the ladder in `_InvokeOllamaCallOnce` needs a fresh probe
# against a resolved per-call base_url.
function _DiscoverModelForHost {
    param([Parameter(Mandatory)] [string]$BaseUrl)
    # Env override wins (parity with `_Discover` above).
    $envOverride = _EnvOr 'OLLAMA_OFFLOAD_MODEL' $null
    if ($envOverride) {
        $Script:DefaultModelSource = 'env'
        return $envOverride
    }
    # PRIMARY: /api/ps — models currently loaded (authoritative "serving now").
    try {
        $ps = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/ps" -TimeoutSec 5
        if ($ps -and $ps.models -and @($ps.models).Count -gt 0) {
            $Script:DefaultModelSource = 'ps'
            return $ps.models[0].name
        }
    } catch { }
    # SECONDARY: /api/tags — all installed models. First-in-list heuristic
    # (Python picks by size; PS stays with the current /api/ps first-item
    # heuristic for parity with _Discover's existing behavior).
    try {
        $tags = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/tags" -TimeoutSec 5
        if ($tags -and $tags.models -and @($tags.models).Count -gt 0) {
            $Script:DefaultModelSource = 'tags'
            return $tags.models[0].name
        }
    } catch { }
    # Everything failed — fall back to config pin, then hardcoded hint.
    $cfg = Get-OllamaConfig
    $cfgPin = $cfg['model']
    if ($cfgPin) {
        $Script:DefaultModelSource = 'cfg'
        return $cfgPin
    }
    $Script:DefaultModelSource = 'hint'
    return $Script:DefaultModelHint
}

function _DiscoverContextTokensForHost {
    # F4: context tokens keyed to (model, base_url) tuple, not `model` alone.
    # A same-model fleet (host A + host B both serving qwen3:8b at different
    # num_ctx) MUST NOT reuse the module-active host's context on the wire
    # to the override host — that is the silent over-fill this fix closes.
    param(
        [Parameter(Mandatory)] [string]$Model,
        [Parameter(Mandatory)] [string]$BaseUrl
    )
    $cfg = Get-OllamaConfig
    $fallbk = [int]($cfg['context_tokens_fallback'] ?? 8192)

    # Env override.
    $envCtx = _EnvOr 'OLLAMA_OFFLOAD_CONTEXT_TOKENS' $null
    if ($envCtx) {
        try { return [int]$envCtx } catch { }
    }

    # PRIMARY: /api/ps — the LOADED runtime context (what actually gets served).
    try {
        $ps = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/ps" -TimeoutSec 5
        if ($null -ne $ps -and $ps.PSObject.Properties.Name -contains 'models') {
            foreach ($m in $ps.models) {
                if (($m.name -eq $Model -or $m.model -eq $Model) -and
                    $m.PSObject.Properties.Name -contains 'context_length') {
                    return [int]$m.context_length
                }
            }
        }
    } catch { }

    # FALLBACK: /api/show num_ctx from parameters text.
    try {
        $show = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/show" `
                    -ContentType 'application/json' `
                    -Body (@{ model = $Model } | ConvertTo-Json) -TimeoutSec 5
        $paramText = if ($show.PSObject.Properties.Name -contains 'parameters') { $show.parameters } else { $null }
        if ($paramText -is [string] -and $paramText) {
            $m = [regex]::Match($paramText, '(?m)^\s*num_ctx\s+(\d+)\s*$')
            if ($m.Success) { return [int]$m.Groups[1].Value }
        }
    } catch { }

    return $fallbk
}

# ---- P1 — Byte budget derived from live context, not literal ---------------
function Get-OllamaContextBytes {
    [CmdletBinding()] param(
        [Parameter(Mandatory)] [double]$Ratio,
        [int]$CharsPerToken = 4
    )
    if ($Ratio -le 0 -or $Ratio -gt 1) {
        throw [System.ArgumentException]::new("Ratio must be in (0, 1]; got $Ratio")
    }
    $d = _Discover
    return [int]($d.ContextTokens * $Ratio * $CharsPerToken)
}

# ---- P3 — Cooldown sentinel (SHARED with Python) ---------------------------
function _CooldownRemaining {
    # LEGACY: reads the GLOBAL sidecar only. Retained for backward-compat;
    # new code paths call _CooldownRemainingFor to key on host name.
    if (-not (Test-Path $Script:CooldownPath)) { return 0 }
    try {
        $until = [double](Get-Content -Raw $Script:CooldownPath).Trim()
        $now   = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $rem   = [int]($until - $now)
        if ($rem -le 0) {
            Remove-Item $Script:CooldownPath -ErrorAction SilentlyContinue
            return 0
        }
        return $rem
    } catch { return 0 }
}

function _SetCooldown {
    # LEGACY: writes the GLOBAL sidecar only. Retained for backward-compat;
    # new code paths call _SetCooldownFor to key on host name.
    param([int]$Seconds)
    $until = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + $Seconds
    Set-Content -Path $Script:CooldownPath -Value ([string]$until) -Encoding utf8
}

function _SanitizeHostForPath {
    # Twin of Python _sanitize_host_for_path — preserves alphanumeric +
    # hyphen + underscore; other bytes become underscore. Prevents path
    # traversal + OS-illegal chars (colon on Windows, slash, backslash).
    # db-sme Batch-2 NTH#4 fold: empty sanitize -> '_unnamed_' so an
    # empty name never yields 'ollama_offload_cooldown..until' (double-dot).
    param([Parameter(Mandatory)] [string]$Name)
    $sanitized = -join ($Name.ToCharArray() | ForEach-Object {
        if ($_ -match '[A-Za-z0-9_-]') { $_ } else { '_' }
    })
    if ([string]::IsNullOrEmpty($sanitized)) { return '_unnamed_' }
    return $sanitized
}

function _CooldownPathFor {
    # Per-host sidecar path: $env:TEMP/ollama_offload_cooldown.<host>.until.
    # Distinct per host so benching hosts.secondary does NOT bench hosts.primary.
    # Twin of Python _cooldown_path_for.
    param([Parameter(Mandatory)] [string]$HostName)
    $safe = _SanitizeHostForPath -Name $HostName
    return (Join-Path $env:TEMP "ollama_offload_cooldown.$safe.until")
}

function _CooldownRemainingFor {
    # Backward-compat OR: reads BOTH the per-host
    # sidecar AND the legacy GLOBAL sidecar; either being in the future
    # counts. Returns the LONGER remaining. Twin of Python _in_cooldown_for.
    param([Parameter(Mandatory)] [string]$HostName)
    $perHostPath = _CooldownPathFor -HostName $HostName
    $maxRem = 0
    $now    = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    foreach ($p in @($perHostPath, $Script:CooldownPath)) {
        if (-not (Test-Path $p)) { continue }
        try {
            $until = [double](Get-Content -Raw $p).Trim()
            $rem = [int]($until - $now)
            if ($rem -gt 0) {
                if ($rem -gt $maxRem) { $maxRem = $rem }
            }
            else {
                # db-sme Batch-2 NTH#3 fold: unlink expired sidecar (mirror
                # of the legacy _CooldownRemaining cleanup). Per-host sidecars
                # otherwise accumulate one file per host name ever benched.
                Remove-Item $p -ErrorAction SilentlyContinue
            }
        } catch { }
    }
    return $maxRem
}

function _SetCooldownFor {
    # Arm the per-host cooldown — benches ONE host,
    # leaves the others callable. Twin of Python _set_cooldown_for.
    param(
        [Parameter(Mandatory)] [string]$HostName,
        [Parameter(Mandatory)] [int]$Seconds
    )
    $until = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + $Seconds
    $path  = _CooldownPathFor -HostName $HostName
    Set-Content -Path $path -Value ([string]$until) -Encoding utf8
}

# ---- P3 + P4 — the ONE call surface ---------------------------------------
function _InvokeOllamaCallOnce {
    <#
    .SYNOPSIS
      Single-shot Ollama call — one host, no failover. Twin of Python
      _call_ollama_once. The public Invoke-OllamaCall orchestrates failover
      across hosts by catching OllamaUnavailable from this helper and trying
      the next host in the failover chain.
    .PARAMETER TaskPrompt
      The full prompt string.
    .PARAMETER Schema
      Optional hashtable = JSON schema. When set, Ollama enforces server-side (P4).
    .PARAMETER System
      Optional system message.
    .OUTPUTS
      When -Schema is set: PSCustomObject/hashtable parsed from JSON.
      When -Schema is not set: [string] raw content.
    .NOTES
      Throws [OllamaUnavailable] if cooldown is active or the host is unreachable.
      Throws [OllamaCallError] on non-recoverable call failure after retries.
    #>
    [CmdletBinding()] param(
        [Parameter(Mandatory)] [string]$TaskPrompt,
        [hashtable]$Schema,
        [string]$System,
        # PER-CALL TEMPERATURE OVERRIDE. Untyped-with-$null-default
        # rather than [double] on purpose: [double] defaults to 0, which is
        # indistinguishable from a caller deliberately pinning 0 and would silently
        # override the configured default on EVERY call that omitted the parameter.
        # $null means "not supplied" and falls through to _ResolveDefaultTemperature.
        $Temperature = $null,
        # ---- PARITY BLOCK ------------------------------------------
        # Python call_ollama exposed 14 parameters; this twin exposed 8. Enumerated gap:
        # Model, TimeoutSec, Think, KeepAlive, Logprobs, TopLogprobs. All six added here.
        # $null defaults throughout so "not supplied" stays distinguishable from an
        # explicit value — the same reason -Temperature is untyped (a [bool] Think would
        # default to $false and silently send think:false on every call).
        [string]$Model,
        [int]$TimeoutSec = 0,          # 0 = use module/config default
        $Think = $null,                # top-level Ollama field, NOT nested in options
        $KeepAlive = $null,
        [switch]$Logprobs,
        [int]$TopLogprobs = 5,
        # per-call host override. When set, this call
        # resolves the named host via _ResolveHostByName + uses THAT host's
        # url/model for THIS invocation only — bypasses the module-cached
        # $Script:ActiveHost. Unknown name raises OllamaConfigError (typo trap
        # symmetric with the OLLAMA_OFFLOAD_HOST env-driven path). Twin of Python
        # `call_ollama(host=<name>)`. `-HostName` (not `-Host`) avoids clash
        # with the PowerShell automatic `$Host` variable.
        [string]$HostName
    )

    # Per-host cooldown gate — resolved AFTER _ResolveHostByName below
    # (fold sequence: typo trap first, then cooldown check on the
    # RESOLVED host). Kept as a placeholder; the actual check runs post-
    # resolution to key on the correct host name.

    $cfg          = Get-OllamaConfig
    $maxRetries   = [int](_EnvOr 'OLLAMA_OFFLOAD_MAX_RETRIES'   ($cfg['max_retries']   ?? 3))
    $baseBackoff  = [double](_EnvOr 'OLLAMA_OFFLOAD_BASE_BACKOFF_S' ($cfg['base_backoff_s'] ?? 1.0))
    # Config-key parity with ollama_offload_config.json + ollama_offload.py:
    #   cooldown_on_fail_s (JSON L19, py L297) — NOT `cooldown_s`
    #   timeout_s          (JSON L15, py L102) — NOT `call_timeout_s`
    # The previous names silently fell through to hardcoded defaults; config tuning
    # was inert. Env-var names (OLLAMA_OFFLOAD_COOLDOWN_S / OLLAMA_OFFLOAD_CALL_TIMEOUT_S) are kept
    # as-declared in this module's docstring.
    $cooldownOnFail = [int](_EnvOr 'OLLAMA_OFFLOAD_COOLDOWN_S' ($cfg['cooldown_on_fail_s'] ?? 300))
    $callTimeout  = [int](_EnvOr 'OLLAMA_OFFLOAD_CALL_TIMEOUT_S' ($cfg['timeout_s'] ?? 180))

    # Model resolution ladder (F1-F5, Python parity `_call_ollama_once`:711).
    # Precedence (highest first):
    #   1. -Model per-call arg — caller's explicit intent
    #   2. OLLAMA_OFFLOAD_MODEL env — ops override / CI pin (F2)
    #   3. per-host config pin `hosts.<name>.model`
    #   4. per-host runtime discovery (F3) — probes the RESOLVED host, not
    #      the module-active one, so a `-HostName` escape hatch or failover
    #      does not receive the active host's discovered model
    #   5. F5 hint fallback: when active discovery fell to hint, retry against
    #      per-call base_url; if still hint, raise OllamaCallError
    #   6. active-host default from `_Discover` (module cache)
    #
    # Context-tokens F4 identity guard: when the effective model is
    # DEFAULT_MODEL *and* the call host is the module-active host, reuse the
    # cached ContextTokens; otherwise probe (model, base_url) fresh so a
    # same-model fleet at different num_ctx does not over-fill the wire.
    $envModelOverride = _EnvOr 'OLLAMA_OFFLOAD_MODEL' $null
    $keepAlive = '5m'
    $activeHost = _ResolveActiveHost
    if ($HostName) {
        # Per-call host override — skip module `_Discover` (that would probe
        # the WRONG host + block a valid failover on OllamaUnavailable).
        $callHost    = _ResolveHostByName -Name $HostName
        $callChatUrl = $callHost.Url
        $callBase    = ($callChatUrl -replace '/api/[^/]+/?$', '')
        if ($Model) {
            $effectiveModel = $Model
        } elseif ($envModelOverride) {
            $effectiveModel = $envModelOverride
        } elseif ($callHost.Model) {
            $effectiveModel = $callHost.Model
        } else {
            # F3: probe THIS host, not the module-active one. Sets
            # $Script:DefaultModelSource so the F5 retry can see 'hint'.
            $effectiveModel = _DiscoverModelForHost -BaseUrl $callBase
        }
        # F4: context tokens keyed to (model, per-call base_url). Fallback
        # to $callHost.ContextTokensFallback ONLY when the probe returned
        # <= 0 (the helper already applies fallback internally).
        $effectiveContextTokens = _DiscoverContextTokensForHost -Model $effectiveModel -BaseUrl $callBase
        if ($effectiveContextTokens -le 0) { $effectiveContextTokens = [int]$callHost.ContextTokensFallback }
    } else {
        # Active-host path — `_Discover` caches module-scope. Only probe
        # freshly if the ladder chooses a model that _Discover did not pick.
        $d = _Discover     # may throw OllamaUnavailable
        $callHost    = $activeHost
        $callChatUrl = "$($d.BaseUrl)/api/chat"
        $callBase    = $d.BaseUrl
        if ($Model) {
            $effectiveModel = $Model
        } elseif ($envModelOverride) {
            $effectiveModel = $envModelOverride
        } elseif ($Script:DefaultModelSource -eq 'hint') {
            # F5: active-host import-time discovery fell back to the
            # hardcoded hint. Retry against the resolved base before shipping
            # a probably-wrong homelab name. Parity note: the Python twin
            # raises OllamaCallError when the retry ALSO returns 'hint'; the
            # bug-compat mirror follows Python's current behavior — a fix
            # commit is pending upstream and will need re-porting.
            $effectiveModel = _DiscoverModelForHost -BaseUrl $callBase
            if ($Script:DefaultModelSource -eq 'hint') {
                throw [OllamaCallError]::new(
                    "model discovery fell back to `$Script:DefaultModelHint for host=$($callHost.Name) at $callBase. " +
                    "Set OLLAMA_OFFLOAD_MODEL, add ``model`` to the host entry, or ensure /api/ps + /api/tags " +
                    "answer with an installed model before calling.")
            }
        } else {
            $effectiveModel = $d.Model
        }
        # F4 identity guard: reuse cached ContextTokens only when both model
        # AND host match the active-cache tuple.
        if ($effectiveModel -eq $d.Model) {
            $effectiveContextTokens = [int]$d.ContextTokens
        } else {
            $effectiveContextTokens = _DiscoverContextTokensForHost -Model $effectiveModel -BaseUrl $callBase
        }
        $keepAlive = $d.KeepAlive
    }

    # per-host cooldown check keyed on the RESOLVED
    # host. Reads the per-host sidecar OR the legacy GLOBAL sidecar (backward-
    # compat "all benched" signal). A sibling host in cooldown does NOT block
    # us — the entire point of per-host scoping. Twin of Python
    # _in_cooldown_for at call_ollama (fold).
    $perHostCooldown = _CooldownRemainingFor -HostName $callHost.Name
    if ($perHostCooldown -gt 0) {
        throw [OllamaUnavailable]::new(
            "Ollama host=$($callHost.Name) in cooldown for ${perHostCooldown}s more " +
            "(per-host sidecar: $(_CooldownPathFor -HostName $callHost.Name); " +
            "legacy global: $($Script:CooldownPath))"
        )
    }

    $messages = @()
    if ($System) { $messages += @{ role = 'system'; content = $System } }
    $messages += @{ role = 'user'; content = $TaskPrompt }

    # Parity: `temperature` joins `num_ctx` here. Previously this twin sent
    # num_ctx ONLY and Python sent temperature ONLY, so the same logical call produced a
    # different wire body per language. Both keys now travel together in both twins.
    $effectiveTemperature = if ($null -ne $Temperature) { [double]$Temperature }
                            else { _ResolveDefaultTemperature }
    # Parity: this twin computed $keepAlive and then never put it in the body, so the
    # server applied its own default while the Python twin pinned it. Set keep_alive
    # explicitly: a large model can occupy most of a GPU's memory, so a short TTL evicts
    # it between calls and the next call pays a full reload -- which looks like a slow
    # model rather than an eviction. Precedence mirrors Python's
    # _resolve_default_keep_alive:
    #   OLLAMA_OFFLOAD_KEEP_ALIVE -> config.keep_alive -> -1 (resident)
    # NOT the '5m' reported by /api/show: that is the model's advertised default, not an
    # instruction to the client, and treating it as one is how the eviction happens.
    $effectiveKeepAlive = _EnvOr 'OLLAMA_OFFLOAD_KEEP_ALIVE' $null
    if ($null -eq $effectiveKeepAlive -or $effectiveKeepAlive -eq '') {
        $cfgKA = (Get-OllamaConfig)
        $effectiveKeepAlive = if ($null -ne $cfgKA -and $null -ne $cfgKA.keep_alive) { $cfgKA.keep_alive } else { -1 }
    }
    # Parity: per-call -Model and -KeepAlive now override the resolved defaults,
    # matching Python's model= / keep_alive= kwargs.
    if ($Model) { $effectiveModel = $Model }
    if ($null -ne $KeepAlive -and $KeepAlive -ne '') { $effectiveKeepAlive = $KeepAlive }

    if ($TimeoutSec -gt 0) { $callTimeout = $TimeoutSec }   # per-call override (Python timeout_s)

    $body = @{
        model      = $effectiveModel
        messages   = $messages
        stream     = $false
        keep_alive = $effectiveKeepAlive
        options    = @{
            num_ctx     = $effectiveContextTokens
            temperature = $effectiveTemperature
        }
    }
    if ($Schema) { $body.format = $Schema }

    # `think` is a TOP-LEVEL field on Ollama /api/chat — a sibling of stream/format/options,
    # NOT nested inside options. The Python twin records that an in-options placement was a
    # proven no-op, so this twin must not repeat it. Guarded by $null so the default case
    # sends no field at all and the wire payload is unchanged for every existing caller.
    if ($null -ne $Think) { $body.think = [bool]$Think }

    # logprobs: Ollama returns token-level probabilities only when asked. Requested at the
    # top level alongside think; absence in the RESPONSE therefore means "not requested or
    # unsupported by this build", never an empty-but-real distribution.
    if ($Logprobs) {
        $body.logprobs = $true
        if ($TopLogprobs -gt 0) { $body.top_logprobs = $TopLogprobs }
    }

    $lastErr = $null
    # Both twins report elapsed_s, eval_tokens and prompt_tokens in _meta. The token
    # counts come off the same response object the call already receives, so there is no
    # extra request; the elapsed time is a stopwatch around the retry loop.
    $callSw = [System.Diagnostics.Stopwatch]::StartNew()
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        try {
            $resp = Invoke-RestMethod -Method Post -Uri $callChatUrl `
                        -ContentType 'application/json' `
                        -Body ($body | ConvertTo-Json -Depth 12 -Compress) `
                        -TimeoutSec $callTimeout
            $content = $resp.message.content
            if ($Schema) {
                try {
                    $parsed = ($content | ConvertFrom-Json -AsHashtable)
                    # eval_count / prompt_eval_count come straight off $resp. The property
                    # probes are StrictMode-safe on purpose: under Set-StrictMode -Version
                    # Latest, reading a field an older server did not send THROWS, where
                    # Python's .get() would quietly return None. That asymmetry is the
                    # sharpest form of twin drift, and it is why parity is tested rather
                    # than assumed.
                    $evalTok = if ($resp.PSObject.Properties.Name -contains 'eval_count') { $resp.eval_count } else { $null }
                    $promptTok = if ($resp.PSObject.Properties.Name -contains 'prompt_eval_count') { $resp.prompt_eval_count } else { $null }
                    $parsed['_meta'] = @{
                        model         = $effectiveModel
                        active_host   = $callHost.Name
                        elapsed_s     = [math]::Round($callSw.Elapsed.TotalSeconds, 2)
                        eval_tokens   = $evalTok
                        prompt_tokens = $promptTok
                    }
                    return $parsed
                }
                catch {
                    # The host answered; the MODEL produced content that will not parse
                    # or does not carry the required keys. That is a model fault, so it
                    # raises OllamaCallError -- NOT OllamaUnavailable -- and deliberately
                    # does not arm the cooldown: the endpoint is healthy and benching it
                    # would punish every other consumer pointed at it.
                    #
                    # This previously retried and then fell out of the loop returning
                    # $null, where the Python twin raised immediately. A silent $null is
                    # the worst of both: the caller cannot tell a refusal from a success
                    # with no data. Parity with Python's behaviour is the contract.
                    throw [OllamaCallError]::new(
                        "Ollama content did not satisfy the schema: $($_.Exception.Message)")
                }
            }
            return $content
        } catch [OllamaCallError] {
            # A model fault raised by the schema branch above. Re-throw it
            # unchanged: no retry, no cooldown. Only transport failures reach
            # the handler below, and only they justify benching the host.
            throw
        } catch {
            $lastErr = $_.Exception.Message
            if ($attempt -lt $maxRetries) {
                $delay = [int]($baseBackoff * [Math]::Pow(2, $attempt - 1))
                Start-Sleep -Seconds $delay
                continue
            }
            # exhausted retries — arm the PER-HOST
            # cooldown so subsequent calls targeting THIS host no-op cleanly
            # while sibling hosts remain callable. Python parity:
            # _set_cooldown_for(_call_host['name'], ...).
            _SetCooldownFor -HostName $callHost.Name -Seconds $cooldownOnFail
            throw [OllamaUnavailable]::new("Ollama host=$($callHost.Name) call failed after $maxRetries attempts: $lastErr; per-host cooldown engaged")
        }
    }
}

$Script:RotationCounter = 0
$Script:RotationSidecarPath = Join-Path $env:TEMP 'ollama_offload_rotation.next'
$Script:ValidStrategies = @('primary', 'round_robin', 'weighted')

function _LoadRotationCounter {
    # twin of Python _load_rotation_counter.
    # Returns the persisted int or $null on absent / malformed / unreadable.
    if (-not (Test-Path $Script:RotationSidecarPath)) { return $null }
    try {
        return [int](Get-Content -Raw $Script:RotationSidecarPath).Trim()
    } catch { return $null }
}

function _PersistRotationCounter {
    # Twin of Python _persist_rotation_counter. Fail-silent on IO error.
    param([Parameter(Mandatory)] [int]$Value)
    try {
        Set-Content -Path $Script:RotationSidecarPath -Value ([string]$Value) -Encoding utf8
    } catch { }
}
$Script:BadStrategySeen = @{}

function _WarnOnceBadStrategy {
    # hipaa-sme Batch-3 NTH-1 fold — twin of Python _warn_once_bad_strategy.
    # One-shot Write-Warning per unknown strategy value seen this process.
    param([string]$Value)
    if ($Script:BadStrategySeen.ContainsKey($Value)) { return }
    $Script:BadStrategySeen[$Value] = $true
    $valid = ($Script:ValidStrategies -join ', ')
    Write-Warning "ollama_offload: unknown strategy '$Value', defaulting to 'primary' (valid: $valid)"
}

function _BuildRotationPool {
    # twin of Python _build_rotation_pool.
    # 'round_robin' yields each host name once (sorted). 'weighted' yields
    # each host `entry.weight` times (default 1). 'primary' returns empty.
    param([Parameter(Mandatory)] [string]$Strategy)
    $cfg      = Get-OllamaConfig
    $hostsMap = $cfg['hosts'] ?? @{}
    $names    = @($hostsMap.Keys | Sort-Object)
    if ($Strategy -eq 'round_robin') { return $names }
    if ($Strategy -eq 'weighted') {
        $pool = @()
        foreach ($n in $names) {
            $entry = $hostsMap[$n] ?? @{}
            $w = 1
            try {
                $rawWeight = $entry['weight']
                if ($null -ne $rawWeight) { $w = [Math]::Max(1, [int]$rawWeight) }
            } catch { $w = 1 }
            for ($i = 0; $i -lt $w; $i++) { $pool += $n }
        }
        return $pool
    }
    return @()
}

function _ResolveConsumerHost {
    # Twin of Python _resolve_consumer_host. Returns
    # the affinity host name for the named consumer if configured; otherwise
    # $null. Absent consumer sections are LEGAL; unknown host names are
    # caught by _ResolveHostByName downstream and surface as OllamaConfigError.
    # db-sme Batch-3 NTH-2 fold: type-guard the host field. PowerShell
    # implicit-stringifies non-string values (int, array), and a stringified
    # 'System.Object[]' would then bubble to OllamaConfigError — diverging
    # from the Python twin's silent None-fallthrough. Both are fail-closed
    # but the surface differed; the guard makes them symmetric.
    param([Parameter(Mandatory)] [string]$Consumer)
    $cfg = Get-OllamaConfig
    $section = $cfg[$Consumer]
    if (-not $section -or -not ($section -is [hashtable])) { return $null }
    $h = $section['host']
    if (-not ($h -is [string])) { return $null }
    if ([string]::IsNullOrWhiteSpace($h)) { return $null }
    return $h
}

function _NextHostByStrategy {
    # twin of Python _next_host_by_strategy.
    # Returns $null for 'primary' or empty pool; otherwise advances
    # $Script:RotationCounter modulo pool size, skipping cooldown'd hosts.
    # When every candidate is benched, returns pool[0] anyway so the caller
    # can propagate a clean OllamaUnavailable rather than looping forever.
    # db-sme Batch-3 NTH-1 fold: guard against Int32 overflow — PS `++` at
    # Int32.MaxValue silently wraps to MinValue giving a NEGATIVE index into
    # the pool (which PS legally treats as a from-end index — order flip).
    # Reset to 0 when nearing the ceiling. Python twin uses arbitrary-
    # precision int so this is a PS-only guard.
    param([Parameter(Mandatory)] [string]$Strategy)
    $pool = _BuildRotationPool -Strategy $Strategy
    if (-not $pool -or $pool.Count -eq 0) { return $null }
    if ($Script:RotationCounter -ge ([int]::MaxValue - 1000)) {
        $Script:RotationCounter = 0
    }
    # load persisted counter (cross-language sync) so a mixed
    # Py+PS caller sees a SHARED cursor. Fail-open on read errors.
    $persisted = _LoadRotationCounter
    if ($null -ne $persisted) { $Script:RotationCounter = $persisted }
    $n = $pool.Count
    for ($i = 0; $i -lt $n; $i++) {
        $idx = $Script:RotationCounter % $n
        $Script:RotationCounter++
        _PersistRotationCounter -Value $Script:RotationCounter
        $candidate = $pool[$idx]
        if ((_CooldownRemainingFor -HostName $candidate) -eq 0) { return $candidate }
    }
    return $pool[0]
}

function Invoke-OllamaCall {
    <#
    .SYNOPSIS
      Call Ollama with optional cross-host failover.
    .DESCRIPTION
      Twin of Python `call_ollama`. When -FailoverHosts (or
      `config.failover_hosts`) is set, iterates a chain
      [primary, ...failoverHosts] with duplicates removed and calls
      _InvokeOllamaCallOnce for each. A OllamaUnavailable from one host causes
      the orchestrator to try the next; a OllamaConfigError (typo trap)
      bubbles immediately.
    .PARAMETER TaskPrompt
      The full prompt string.
    .PARAMETER Schema
      Optional hashtable = JSON schema. When set, Ollama enforces server-side (P4).
    .PARAMETER System
      Optional system message.
    .PARAMETER HostName
      Pin the PRIMARY host for this call. Unknown name raises
      OllamaConfigError. When empty, the module-resolved active host is used.
    .PARAMETER FailoverHosts
      Ordered list of host names to try if the primary fails. When empty,
      sources from `config.failover_hosts` (empty list means "no failover"
      = single-host behaviour).
    .PARAMETER Strategy
      Rotation strategy for primary-host selection.
      One of 'primary' (default; single-host behaviour), 'round_robin' (each
      configured host in turn), or 'weighted' (each host repeated
      hosts.<name>.weight times, default 1). Per-call value wins over
      config.strategy. Only fires when -HostName + -Consumer do not already
      pin one. Rotation counter is process-scope; twin counters
      (Python + PS) are independent by design — see the multi-host precedence rules in the README
      drawer for cross-twin drift semantics.
    .PARAMETER Consumer
      Name of a consumer section (e.g. 'summarizer', 'classifier') —
      Per-consumer host affinity. When the section carries a `host`
      field, that value becomes the primary. Wins over -Strategy but loses
      to explicit -HostName. A consumer-pinned host still respects cooldown;
      pair with -FailoverHosts for resilience.
    .OUTPUTS
      Same shape as _InvokeOllamaCallOnce. When failover triggered (a
      non-primary host served), the returned hashtable (schema mode) grows
      _meta.failover_chain listing the hosts tried in order.
    .NOTES
      Throws [OllamaConfigError] if any name in the chain is unknown.
      Throws [OllamaUnavailable] if every host in the chain fails.
    #>
    [CmdletBinding()] param(
        [Parameter(Mandatory)] [string]$TaskPrompt,
        [hashtable]$Schema,
        [string]$System,
        [string]$HostName,
        [string[]]$FailoverHosts,
        # Per-call temperature override. Forwarded to EVERY host in the
        # failover chain: a retry that silently changed sampling would not be comparable
        # to the attempt it replaced. $null = fall through to _ResolveDefaultTemperature.
        $Temperature = $null,
        # rotation strategy for primary host selection.
        # 'primary' (default) uses the module-resolved active host. 'round_robin'
        # rotates across all configured hosts. 'weighted' rotates by
        # hosts.<name>.weight (default 1). Only fires when -HostName is not
        # pinned. Per-call value wins over config.strategy.
        [ValidateSet('', 'primary', 'round_robin', 'weighted')]
        [string]$Strategy = '',
        # per-consumer host affinity. Name of a
        # consumer section (e.g. 'summarizer', 'classifier'). When
        # set + the section carries a `host` field, that becomes the primary
        # unless -HostName is pinned. Twin of Python call_ollama(consumer=<name>).
        [string]$Consumer
    ,
        # Parity: mirrors _InvokeOllamaCallOnce and Python call_ollama.
        # All six are FORWARDED to every host in the failover chain: a retry that
        # silently dropped -Think or -Logprobs would not be comparable to the attempt
        # it replaced.
        [string]$Model,
        [int]$TimeoutSec = 0,
        $Think = $null,
        $KeepAlive = $null,
        [switch]$Logprobs,
        [int]$TopLogprobs = 5)

    $cfg = Get-OllamaConfig
    if ($null -eq $FailoverHosts) {
        $configFailover = $cfg['failover_hosts']
        $FailoverHosts = @(if ($null -ne $configFailover) { $configFailover })
    }

    # per-consumer host affinity. Runs BEFORE strategy
    # so a consumer-pinned host is not overwritten by round-robin/weighted.
    if (-not $HostName -and $Consumer) {
        $consumerHost = _ResolveConsumerHost -Consumer $Consumer
        if ($consumerHost) { $HostName = $consumerHost }
    }

    # Strategy resolution — per-call wins, then config, then default.
    $effectiveStrategy = if ($Strategy) { $Strategy } else { ($cfg['strategy'] ?? 'primary') }
    if ($Script:ValidStrategies -notcontains $effectiveStrategy) {
        # hipaa-sme Batch-3 NTH-1 fold: warn once/process on unknown strategy.
        _WarnOnceBadStrategy -Value $effectiveStrategy
        $effectiveStrategy = 'primary'
    }
    if (-not $HostName -and $effectiveStrategy -in @('round_robin', 'weighted')) {
        $picked = _NextHostByStrategy -Strategy $effectiveStrategy
        if ($picked) { $HostName = $picked }
    }

    if (-not $FailoverHosts -or $FailoverHosts.Count -eq 0) {
        # Single-host behaviour — one host, one try.
        if ($HostName) {
            return _InvokeOllamaCallOnce -TaskPrompt $TaskPrompt -Schema $Schema -System $System -Temperature $Temperature -HostName $HostName -Model $Model -TimeoutSec $TimeoutSec -Think $Think -KeepAlive $KeepAlive -Logprobs:$Logprobs -TopLogprobs $TopLogprobs
        }
        return _InvokeOllamaCallOnce -TaskPrompt $TaskPrompt -Schema $Schema -System $System -Temperature $Temperature -Model $Model -TimeoutSec $TimeoutSec -Think $Think -KeepAlive $KeepAlive -Logprobs:$Logprobs -TopLogprobs $TopLogprobs
    }

    # Failover path — build ordered chain [primary, ...failover] dedup'd.
    $primaryName = if ($HostName) { $HostName } else { (_ResolveActiveHost).Name }
    $chain = @($primaryName)
    foreach ($fh in $FailoverHosts) {
        if ($chain -notcontains $fh) { $chain += $fh }
    }

    $lastErr = $null
    for ($i = 0; $i -lt $chain.Count; $i++) {
        $h = $chain[$i]
        try {
            $result = _InvokeOllamaCallOnce -TaskPrompt $TaskPrompt -Schema $Schema -System $System -Temperature $Temperature -HostName $h -Model $Model -TimeoutSec $TimeoutSec -Think $Think -KeepAlive $KeepAlive -Logprobs:$Logprobs -TopLogprobs $TopLogprobs
            if ($Schema -and $i -gt 0 -and $result -is [hashtable]) {
                # Failover fired — record the actual traversal for audit.
                if (-not $result.ContainsKey('_meta')) { $result['_meta'] = @{} }
                $result['_meta']['failover_chain'] = @($chain[0..$i])
            }
            return $result
        }
        catch [OllamaUnavailable] {
            $lastErr = $_.Exception.Message
            continue
        }
    }
    throw [OllamaUnavailable]::new("All $($chain.Count) hosts exhausted (chain=$($chain -join ', ')); last error: $lastErr")
}

function Get-OllamaHealthcheck {
    <#
    .SYNOPSIS
      Twin of Python `ollama_healthcheck` — probe /api/tags on EVERY host in
      ollama_offload_config.json:hosts. Strictly informational; no cooldown
      side-effects, no auto-routing.
    .PARAMETER TimeoutSec
      Per-probe timeout in seconds (default 4).
    .OUTPUTS
      Array of hashtables, sorted by host name for deterministic ordering:
        @{ Name; Url; Reachable; LatencyMs; ModelCount; Error }
      Reachable=$false + Error=<string> when the probe fails; the loop
      continues past a bad host.
    .NOTES
      Landed with the multi-host work. Failover / preferred-alternate selection
      landed alongside failover; healthcheck itself remains
      informational and does not auto-route.
    #>
    [CmdletBinding()] param([int]$TimeoutSec = 4)
    $cfg      = Get-OllamaConfig
    $hostsMap = $cfg['hosts'] ?? @{}
    $results  = @()
    foreach ($name in ($hostsMap.Keys | Sort-Object)) {
        $entry   = $hostsMap[$name] ?? @{}
        $chatUrl = $entry['url'] ?? $cfg['url'] ?? 'http://localhost:11434/api/chat'
        $base    = ($chatUrl -replace '/api/[^/]+/?$', '')
        $tagsUrl = "$base/api/tags"
        $sw      = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $data = Invoke-RestMethod -Method Get -Uri $tagsUrl -TimeoutSec $TimeoutSec
            $sw.Stop()
            $modelCount = if ($null -ne $data.models) { @($data.models).Count } else { 0 }
            $results += @{
                Name       = $name
                Url        = $chatUrl
                Reachable  = $true
                LatencyMs  = [int]$sw.ElapsedMilliseconds
                ModelCount = $modelCount
                Error      = $null
            }
        }
        catch {
            $sw.Stop()
            $msg = $_.Exception.Message
            if ($msg.Length -gt 200) { $msg = $msg.Substring(0, 200) }
            $results += @{
                Name       = $name
                Url        = $chatUrl
                Reachable  = $false
                LatencyMs  = [int]$sw.ElapsedMilliseconds
                ModelCount = $null
                Error      = $msg
            }
        }
    }
    return , $results
}

# ---- Small utility — probe reachability without consuming a real call ------
function Test-OllamaReachable {
    [CmdletBinding()] param()
    if ((_CooldownRemaining) -gt 0) { return $false }
    try {
        $baseUrl = _GetBaseUrl
        Invoke-RestMethod -Method Get -Uri "$baseUrl/api/tags" -TimeoutSec 4 | Out-Null
        return $true
    } catch { return $false }
}


function Test-OllamaProbe {
    <#
    .SYNOPSIS
      Twin of Python `probe_ollama()` — cheap up/down + reason, without making a model call.
    .OUTPUTS
      [pscustomobject] @{ Ok = [bool]; Reason = [string] }.  Python returns (bool, str);
      a named object is the idiomatic PS shape for the same two values.
    .NOTES
      Reports the CLIENT-SIDE cooldown distinctly from unreachability. Those are different
      faults and conflating them is what produced "the host entered a cooldown" when it was
      serving fine and only this workstation was benched.
    #>
    [CmdletBinding()] param()
    try {
        $rem = _CooldownRemaining
        if ($rem -gt 0) {
            return [pscustomobject]@{ Ok=$false; Reason="client-side cooldown active for ${rem}s (local breaker, NOT remote health)" }
        }
    } catch { }
    try {
        if (-not (Test-OllamaReachable)) {
            return [pscustomobject]@{ Ok=$false; Reason="unreachable at $(Get-OllamaActiveHost)" }
        }
    } catch {
        return [pscustomobject]@{ Ok=$false; Reason="probe failed: $($_.Exception.Message)" }
    }
    return [pscustomobject]@{ Ok=$true; Reason="reachable via $(Get-OllamaActiveHost)" }
}

Export-ModuleMember -Function `
    Get-OllamaConfig, Get-OllamaConsumerConfig, Get-OllamaContextBytes, `
    Invoke-OllamaCall, Test-OllamaReachable, Get-OllamaActiveHost, `
    Get-OllamaHealthcheck, Test-OllamaProbe
