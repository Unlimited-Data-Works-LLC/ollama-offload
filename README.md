# ollama-offload

A resilient client for offloading work to a self-hosted [Ollama](https://ollama.com), in
**Python and PowerShell**, with the two kept in deliberate lockstep.

The offload pattern it supports: a local model does the wide, cheap, mechanical pass over many
items, and an expensive model — or a person — keeps the judgment calls. That only pays off if
the cheap pass is dependable, and a bare HTTP call to `/api/chat` is not.

```python
import ollama_offload as oo

schema = {"type": "object", "properties": {"corrective": {"type": "boolean"}},
          "required": ["corrective"]}
result = oo.call_ollama("Is this commit a bug fix?", schema=schema)
# {'corrective': True, '_meta': {'model': ..., 'active_host': ..., 'elapsed_s': 0.4, ...}}
```

```powershell
Import-Module ./src/OllamaOffload.psm1 -Force
$schema = @{ type='object'; properties=@{ corrective=@{ type='boolean' } }; required=@('corrective') }
$result = Invoke-OllamaCall -TaskPrompt 'Is this commit a bug fix?' -Schema $schema
```

## The idea worth stealing: a 200 is not a success

Most clients treat any failure as a failure of the endpoint. That is wrong, and the mistake is
expensive once more than one caller shares a host.

| What happened | Whose fault | What this client does |
|---|---|---|
| Connection refused, timeout, DNS failure | the **host** | retry with backoff, then bench that host for a cooldown period |
| `200 OK` whose body will not parse, or is missing required keys | the **model** | raise immediately; leave the host in service |

A model that answers "I'm afraid I can't do that" where JSON was required has not proved the
endpoint is unhealthy — it has proved this prompt went badly. Putting the endpoint into cooldown
for that penalizes every other consumer pointed at it, and the failure is invisible in the logs
because the request *succeeded*.

The two tests that pin this are named after it, one per language:
`test_non_json_200_does_not_arm_cooldown` and `does not bench the host when a 200 carries
unparseable content`.

## What else it handles

- **Runtime model discovery, per host** — asks the *resolved* host's `/api/ps` what is loaded,
  then `/api/tags`. The discovery helpers accept a kwarg-only `base_url=` so a `-HostName`
  override or a failover probes THAT host, not the module-active one. Nothing about a model name
  is hardcoded on a reachable host (see [Model resolution](#model-resolution)).
- **Runtime context discovery** — reads the model's real context window and sizes input against
  a per-consumer ratio of it. The architectural maximum from `/api/show` is often far larger
  than what is actually loaded under VRAM pressure, so using it over-fills prompts.
- **`keep_alive` that follows the server instead of overriding it** — Ollama decides how long
  a model stays loaded from the `keep_alive` value on each request, for every client of that
  server. Leaving the value out does not mean "leave it as it is": the server applies its own
  short default. So by default (`"auto"`) this client first asks the server what is loaded
  (`/api/ps`). If the model is already set to stay loaded indefinitely, it sends `-1` (as a
  number; the text `"-1"` is rejected) to keep it that way. Otherwise it sends nothing. A model
  name without a tag matches the same name with `:latest`. If the server cannot be asked, the
  client reuses its last answer for that server and model, kept in a small file in the temp
  folder and shared by the Python and PowerShell versions, so one slow request does not unload
  a model that was meant to stay. Earlier versions always sent `-1`, which kept every model they
  touched loaded indefinitely on a shared GPU. A value you set yourself (`-1`, a number of
  seconds, or a duration such as `"30m"`, in the config, an environment variable or the call) is
  always sent as given, and `null` sends nothing.
- **`think: false` by default** — a reasoning model asked for structured extraction spends time
  thinking to no benefit, and can stall for the whole timeout. Measured here: 1.5s with the
  field set versus a hang past 180s without it. Override per call.
- **Per-host cooldown, shared between the twins** — the sidecar lives under `$TEMP`, so a
  cooldown armed by a Python caller is honoured by a PowerShell one. A backed-off endpoint
  should stay backed off regardless of which language noticed.
- **Cross-host failover and rotation** — `primary`, `round_robin` or `weighted` across a
  `hosts` map, with per-consumer host affinity. See [Failover](#failover) for the exception
  contract the orchestrator relies on.
- **Structured output** — your JSON schema goes out as `/api/chat`'s `format` field, and
  required top-level keys are checked on the way back.

## Install

No packaging, no dependencies beyond the standard library. Copy `src/` where you want it.

```bash
git clone https://github.com/Unlimited-Data-Works-LLC/ollama-offload
cp -r ollama-offload/src /path/to/your/project/vendor/ollama-offload
```

Vendoring it read-only? Keep your config outside the dependency and point at it, so an
update never overwrites your hosts:

```bash
export OLLAMA_OFFLOAD_CONFIG=/etc/myapp/ollama-offload.json
```

Python 3.10+ (`jsonschema` optional, for config validation). PowerShell 7+ for the `.psm1`.

## Configuring

`src/ollama_offload_config.json` is the whole surface. A single-host setup needs two keys:

```json
{
  "url": "http://localhost:11434/api/chat",
  "model": "qwen3.6:35b"
}
```

Multi-host adds a map and a default:

```json
{
  "default_host": "primary",
  "hosts": {
    "primary":   { "url": "http://localhost:11434/api/chat", "model": "qwen3.6:35b" },
    "secondary": { "url": "http://ollama-2.internal:11434/api/chat", "model": "qwen3.6:35b" }
  },
  "failover_hosts": ["secondary"],
  "strategy": "primary"
}
```

Omitting `model` on a host entry (or at the top level) is legal and now supported end-to-end:
the client discovers the model live from that host's `/api/ps` on every call. See below.

### Consumers

A *consumer* is a named caller with its own slice of the context window:

```json
{ "summarizer": { "context_usage_ratio": 0.30 },
  "classifier": { "context_usage_ratio": 0.40, "batch_size": 25 } }
```

```python
oo.call_ollama(prompt, schema=schema, consumer="summarizer")
```

Sizing input per caller keeps a chatty consumer from crowding out a careful one, and puts every
caller's budget in one reviewable file instead of scattered across call sites.

### Environment overrides

Every variable is prefixed `OLLAMA_OFFLOAD_`, and that prefix is not decoration. `OLLAMA_HOST`,
`OLLAMA_MODELS` and `OLLAMA_KEEP_ALIVE` belong to **Ollama itself** — a library that read them
would silently change behaviour for anyone who has them set for the server or the CLI. There is
a test asserting this library never reads them.

| Variable | Effect |
|---|---|
| `OLLAMA_OFFLOAD_CONFIG` | read the config from this path instead of next to the module |
| `OLLAMA_OFFLOAD_HOST` | pick a `hosts.<name>` entry |
| `OLLAMA_OFFLOAD_URL` | raw `/api/chat` URL, bypassing the map |
| `OLLAMA_OFFLOAD_MODEL` | override the discovered/pinned model (see [Model resolution](#model-resolution)) |
| `OLLAMA_OFFLOAD_TIMEOUT_S` · `_TEMPERATURE` · `_CONTEXT_TOKENS` | override individual knobs |
| `OLLAMA_OFFLOAD_THINK` · `_KEEP_ALIVE` | override the request fields |

A name given to `OLLAMA_OFFLOAD_HOST` that is not in the map raises `OllamaConfigError` rather
than falling back — a typo that silently routes to the default host is indistinguishable from
success, which is the worst property a routing bug can have.

**Env-value stripping.** `OLLAMA_OFFLOAD_MODEL` is stripped at both read sites (import-time
discovery and the per-call wire ladder). An empty string or a whitespace-only value counts as
**UNSET** — it does not override anything, it falls through to the next tier. A non-empty
stripped value wins over both the config pin and any discovered value. This behavior is
test-locked in `tests/test_ollama_offload.py::TestNonCoincidentModelDiscovery`.

## Model resolution

The client picks a model for every call by walking a ladder, highest precedence first. Any
tier that produces a non-empty value wins; the rest are skipped.

1. **Per-call `model=` / `-Model` argument** — the caller's explicit intent for this one call.
2. **`OLLAMA_OFFLOAD_MODEL` env var** — ops override or CI pin. Stripped on read; empty or
   whitespace-only = UNSET.
3. **Config-file pin** — `hosts.<name>.model` for the resolved host, else top-level `model`.
4. **`/api/ps` on the target host** — the model *currently loaded and serving*, discovered live
   via `_discover_model(base_url=<target>)`. This is per-host: a `-HostName` override or a
   failover probes that host, not the module-active one.
5. **`/api/tags` on the target host** — same discovery helper, if `/api/ps` returned nothing.
6. **`_DEFAULT_MODEL_HINT`** (`qwen3.6:35b`) — **bootstrap only**. This tier fires only at
   *import time* when the primary host is unreachable, so a caller can still construct the
   module without a live Ollama. It **never ships to the wire on a reachable host**: the F5
   guard in `_call_ollama_once` raises `OllamaUnavailable` rather than send a hint to a host
   that answered a probe with no models.

Both versions now follow this order in the same way. Before October 2026, the PowerShell version
skipped step 3 (a model named in the config) for the default host and used the discovered model
instead. If you use the PowerShell version and your config names a `model`, it now uses that
model, as the Python version always has.

### Recovery after a bootstrap-time miss (F5)

If the primary host was down at import (tier 6 fired and `_DEFAULT_MODEL_SOURCE == "hint"`), a
later call that finds the primary back up **rewrites the module-globals**: `DEFAULT_MODEL` and
`DEFAULT_CONTEXT_TOKENS` are refreshed atomically from the retry's live probe, so every
subsequent call sees the recovered state and not the import-time hint. Without the paired
context-tokens refresh, the `num_ctx` shortcut on the fast path would ship stale import-time
tokens against the newly-discovered model — the `872de70` fix closes that gap.

### Per-host discovery kwargs

`_discover_model(*, base_url=None)` and `_discover_context_tokens(model, *, base_url=None)`
accept a keyword-only `base_url=`. The default (`None`) preserves the module-active probe used
at import; `_call_ollama_once` threads the resolved per-call `base_url` through both on every
call so that the wire body's `model` and `options.num_ctx` describe the *actual* target host,
even under `-HostName` override or failover.

## Budget refusal

Pass a share of the context window and the call refuses, before sending, a prompt that will not fit:

```python
oo.call_ollama(prompt, schema=schema, input_ratio=0.4)   # raises OllamaPromptTooLarge if too big
```
```powershell
Invoke-OllamaCall -TaskPrompt $prompt -Schema $schema -InputRatio 0.4
```

The limit is `input_ratio` x the context window this call will use (Ollama's `num_ctx`, in tokens)
x 4 characters per token. The system prompt counts towards it. Nothing is ever cut short: a
silently shortened prompt loses its end, often the instructions, and the model still answers.
You decide what is safe to trim. The client first checks that the server is reachable and not
in a cooldown, so a server that is down raises `OllamaUnavailable`, never "too large". The
error carries `chars`, `budget`, `ratio`, `context_tokens` and `context_confirmed` (PowerShell:
`Chars`, `Budget`, `Ratio`, `ContextTokens`, `ContextConfirmed`).

## Status

Ask the module what it will do, instead of reading its internals:

```python
oo.status()
# {'host': 'primary', 'base_url': 'http://localhost:11434', 'model': 'qwen3:8b', 'model_source': 'ps',
#  'context_tokens': 8192, 'context_confirmed': True, 'cooldown_s': 0}
oo.clear_cooldown()   # -> the paths of the cooldown files it removed
```
```powershell
Get-OllamaStatus      # same keys
Clear-OllamaCooldown
```

`model` and `context_tokens` are what the next call to the default host would send, following the
same order as a real call (see [Model resolution](#model-resolution)). Asking for the status changes
nothing. `context_confirmed` is true only when the server, asked again, reports that same context
window; false means the number is the configured fallback (or the server has changed), so size
your input conservatively. `model_source` says where the model came from: `env`
(`OLLAMA_OFFLOAD_MODEL`), `cfg` (named in the config), `ps` (loaded on the server), `tags`
(installed but not loaded) or `hint` (nothing could be discovered, so the built-in fallback name).

## Errors

```
OllamaCallError          base for call-time failures (schema-invalid content, model faults)
├── OllamaPromptTooLarge the prompt is over input_ratio x the window this call would send;
│                        raised before sending, nothing cut, no cooldown, no failover
└── OllamaUnavailable    the host could not be reached, is in cooldown, or discovery
                         cannot find a real model on a reachable host — the failover
                         orchestrator catches THIS class only
OllamaConfigError        the configuration is wrong; retrying will not help
```

Catch `OllamaCallError` to sweep both call-time branches. The subclass relationship is part
of the contract, and it is load-bearing: the failover orchestrator catches
`OllamaUnavailable` specifically so that a **model** fault on one host does not silently
route the same broken prompt to the next. Only **transport-shaped** faults (unreachable,
cooldown, hint-only-discovery) propagate to the next host in the chain.

## Failover

`call_ollama` (Python) and `Invoke-OllamaCall` (PowerShell) walk an ordered chain of
`[primary, ...failover_hosts]`, deduplicated. For each host:

1. The per-call `base_url` is threaded through model + context discovery — every host gets its
   own live probe, not the module-active one's cached answers.
2. A **transport fault** (`OllamaUnavailable`) causes the orchestrator to try the next host.
   This includes: connection refused, timeout, per-host cooldown, and the F5 case where
   the host answered a probe but neither `/api/ps` nor `/api/tags` yielded an installed model
   and the caller supplied no `OLLAMA_OFFLOAD_MODEL` / config pin.
3. A **model fault** (`OllamaCallError` but *not* `OllamaUnavailable`) bubbles up unchanged —
   the host served content, so failover to a different host would just repeat the bad prompt.
4. `OllamaConfigError` (bad name, missing config) bubbles up unchanged.

If every host in the chain fails, `OllamaUnavailable` is raised naming the chain that was
tried.

## Testing

Both suites are fully offline against canned fixtures. A test suite that needs a GPU is a
suite nobody runs.

```bash
python3 -m unittest discover -s tests
```
```powershell
Invoke-Pester ./tests
```

The model-resolution contract is locked in
`tests/test_ollama_offload.py::TestNonCoincidentModelDiscovery` — 19 assertions covering:

- **Non-coincident model discovery** — `/api/ps` returns a model that differs from the config
  pin; the discovered model wins and reaches the wire body.
- **Config-pin-wins-over-discovered** — a config pin beats `/api/ps` (tier 3 above tier 4).
- **Whitespace-only env** — `OLLAMA_OFFLOAD_MODEL="   "` counts as UNSET.
- **Empty-string env** — `OLLAMA_OFFLOAD_MODEL=""` counts as UNSET.
- **`_discover_model` direct strip** — the strip happens inside the helper too, not only at
  the wire ladder read site.
- **Mid-run primary recovery** — two-call sequence: primary down at import, primary back up on
  call 2, `DEFAULT_MODEL` is rewritten from the recovered probe.
- **`num_ctx` refresh on recovery** — the paired `DEFAULT_CONTEXT_TOKENS` refresh, without
  which the fast-path `num_ctx == DEFAULT_MODEL` shortcut would ship stale tokens.

## The twins

Both implementations are first-class. PowerShell is not a port that lags behind: they share a
config file and a cooldown sidecar, so a behavioural gap between them is a correctness bug, not
a cosmetic one.

Keeping twins honest requires testing them, because reading them is not enough. A worked example
from this codebase: `/api/show` returns `parameters` as a **newline-delimited string**, not an
object. Python's `.get()` on it quietly returns `None` and carries on. PowerShell under
`Set-StrictMode -Version Latest` *throws*. Identical code shape, divergent behaviour, and only
one side fails loudly — the sharpest form of drift there is, and invisible to code review.

## License

MIT — see [LICENSE](LICENSE).
