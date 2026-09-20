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

- **Runtime model discovery** — asks `/api/ps` what is actually loaded, then `/api/tags`, rather
  than hardcoding a model name that goes stale.
- **Runtime context discovery** — reads the model's real context window and sizes input against
  a per-consumer ratio of it. The architectural maximum from `/api/show` is often far larger
  than what is actually loaded under VRAM pressure, so using it over-fills prompts.
- **Explicit `keep_alive`** — Ollama defaults to unloading after `5m`. On a large model that
  means the next call pays a full reload, which reads as "the model got slow" rather than "the
  model was evicted". This client always sends the field.
- **`think: false` by default** — a reasoning model asked for structured extraction spends time
  thinking to no benefit, and can stall for the whole timeout. Measured here: 1.5s with the
  field set versus a hang past 180s without it. Override per call.
- **Per-host cooldown, shared between the twins** — the sidecar lives under `$TEMP`, so a
  cooldown armed by a Python caller is honoured by a PowerShell one. A backed-off endpoint
  should stay backed off regardless of which language noticed.
- **Cross-host failover and rotation** — `primary`, `round_robin` or `weighted` across a
  `hosts` map, with per-consumer host affinity.
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
  "model": "qwen3:8b"
}
```

Multi-host adds a map and a default:

```json
{
  "default_host": "primary",
  "hosts": {
    "primary":   { "url": "http://localhost:11434/api/chat", "model": "qwen3:8b" },
    "secondary": { "url": "http://ollama-2.internal:11434/api/chat", "model": "qwen3:8b" }
  },
  "failover_hosts": ["secondary"],
  "strategy": "primary"
}
```

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
| `OLLAMA_OFFLOAD_MODEL` | override the discovered model |
| `OLLAMA_OFFLOAD_TIMEOUT_S` · `_TEMPERATURE` · `_CONTEXT_TOKENS` | override individual knobs |
| `OLLAMA_OFFLOAD_THINK` · `_KEEP_ALIVE` | override the request fields |

A name given to `OLLAMA_OFFLOAD_HOST` that is not in the map raises `OllamaConfigError` rather
than falling back — a typo that silently routes to the default host is indistinguishable from
success, which is the worst property a routing bug can have.

## Errors

```
OllamaCallError          the call completed and the answer was unusable (model fault)
└── OllamaUnavailable    the host could not be reached, or is in cooldown (transport fault)
OllamaConfigError        the configuration is wrong; retrying will not help
```

Catch `OllamaCallError` to sweep both call-time branches. The subclass relationship is part of
the contract.

## The twins

Both implementations are first-class. PowerShell is not a port that lags behind: they share a
config file and a cooldown sidecar, so a behavioural gap between them is a correctness bug, not
a cosmetic one.

Keeping twins honest requires testing them, because reading them is not enough. A worked example
from this codebase: `/api/show` returns `parameters` as a **newline-delimited string**, not an
object. Python's `.get()` on it quietly returns `None` and carries on. PowerShell under
`Set-StrictMode -Version Latest` *throws*. Identical code shape, divergent behaviour, and only
one side fails loudly — the sharpest form of drift there is, and invisible to code review.

The suites mirror each other case for case, and both run fully offline against canned fixtures.
A test suite that needs a GPU is a suite nobody runs.

```bash
python3 -m unittest discover -s tests      # 8 tests
```
```powershell
Invoke-Pester ./tests                      # 7 tests
```

## License

MIT — see [LICENSE](LICENSE).
