<#
    Offline tests for the PowerShell twin.

    No Ollama, no GPU, no network: Invoke-RestMethod is mocked and every
    response comes from a canned fixture. Run with:

        Invoke-Pester ./tests

    These mirror the Python suite case for case. That is the point -- the two
    twins share a config file and a cooldown sidecar, so a behavioural gap
    between them is not cosmetic, and the only way to know they agree is to ask
    both the same questions.
#>

BeforeAll {
    $script:ModulePath = Join-Path (Split-Path $PSScriptRoot -Parent) 'src/OllamaOffload.psm1'

    # /api/show reports `parameters` as a newline-delimited STRING. Reading it
    # as an object is what broke this twin: under Set-StrictMode -Version
    # Latest a property access on a String THROWS, where Python's .get()
    # quietly returns None. Identical code shape, divergent behaviour, and only
    # one side fails loudly -- so the fixture keeps the real shape.
    $script:Show = [pscustomobject]@{
        parameters = "num_keep                       24`nstop                           `"<|im_end|>`""
        model_info = [pscustomobject]@{ 'qwen3.context_length' = 262144 }
    }
    $script:Ps    = [pscustomobject]@{ models = @([pscustomobject]@{ name = 'qwen3:8b'; context_length = 8192 }) }
    $script:Tags  = [pscustomobject]@{ models = @([pscustomobject]@{ name = 'qwen3:8b' }) }

    function script:New-ChatResponse {
        param([string]$Content, [int]$EvalCount = 11, [int]$PromptEvalCount = 22)
        [pscustomobject]@{
            message           = [pscustomobject]@{ content = $Content }
            eval_count        = $EvalCount
            prompt_eval_count = $PromptEvalCount
        }
    }
}

Describe 'OllamaOffload' {

    BeforeEach {
        # Redirect the cooldown sidecar so tests never touch real state and
        # never observe each other's.
        $script:TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("oo-test-" + [guid]::NewGuid())
        New-Item -ItemType Directory -Path $script:TempDir | Out-Null
        $env:TEMP = $script:TempDir
        $env:OLLAMA_OFFLOAD_URL = 'http://test.invalid:11434/api/chat'
        Import-Module $script:ModulePath -Force
    }

    AfterEach {
        Remove-Module OllamaOffload -Force -ErrorAction SilentlyContinue
        Remove-Item $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item Env:OLLAMA_OFFLOAD_URL -ErrorAction SilentlyContinue
    }

    Context 'the /api/show parameters field is a string' {

        It 'discovers a context window without throwing under StrictMode' {
            $script:Show.parameters | Should -BeOfType [string]
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                throw "no fixture for $Uri"
            }
            { Get-OllamaContextBytes -Ratio 0.4 } | Should -Not -Throw
            Get-OllamaContextBytes -Ratio 0.4 | Should -BeGreaterThan 0
        }
    }

    Context 'the wire body' {

        BeforeEach {
            $script:Sent = [System.Collections.ArrayList]::new()
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') {
                    [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable))
                    return (New-ChatResponse -Content '{"ok":true}')
                }
                throw "no fixture for $Uri"
            }
        }

        It 'sends temperature and num_ctx together' {
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            $body = $script:Sent[-1]
            $body.options.Keys | Should -Contain 'temperature'
            $body.options.Keys | Should -Contain 'num_ctx'
        }

        It 'sends keep_alive explicitly' {
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            $script:Sent[-1].Keys | Should -Contain 'keep_alive'
        }

        It 'emits the same _meta keys as the Python twin' {
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            $result = Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema
            $expected = @('active_host', 'elapsed_s', 'eval_tokens', 'model', 'prompt_tokens')
            ($result['_meta'].Keys | Sort-Object) | Should -Be $expected
            $result['_meta'].eval_tokens   | Should -Be 11
            $result['_meta'].prompt_tokens | Should -Be 22
        }
    }

    Context 'transport faults and model faults are separated' {

        It 'does not bench the host when a 200 carries unparseable content' {
            # The endpoint answered. The MODEL produced prose where JSON was
            # required. Benching a healthy endpoint for that punishes every
            # other consumer pointed at it.
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { return (New-ChatResponse -Content "I'm afraid I can't do that.") }
                throw "no fixture for $Uri"
            }
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            { Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema } | Should -Throw

            $remaining = InModuleScope OllamaOffload { _CooldownRemainingFor (Get-OllamaActiveHost) }
            $remaining | Should -Be 0 -Because 'a model fault must never arm the cooldown'
        }

        It 'benches the host when the transport fails' {
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { throw [System.Net.Http.HttpRequestException]::new('connection refused') }
                throw "no fixture for $Uri"
            }
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            { Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema } | Should -Throw

            $remaining = InModuleScope OllamaOffload { _CooldownRemainingFor (Get-OllamaActiveHost) }
            $remaining | Should -BeGreaterThan 0 -Because 'a transport fault must arm the cooldown'
        }
    }

    Context 'OLLAMA_OFFLOAD_MODEL empty / whitespace is UNSET (c0961a3 i+v)' {

        It 'treats OLLAMA_OFFLOAD_MODEL = "" as UNSET (falls through to /api/ps discovery)' {
            # Parity with Python test_empty_string_env_is_unset. An empty-string
            # env override must NOT ship '' as the model; the ladder must fall
            # through to /api/ps discovery.
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') {
                    [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable))
                    return (New-ChatResponse -Content '{"ok":true}')
                }
                throw "no fixture for $Uri"
            }
            $script:Sent = [System.Collections.ArrayList]::new()
            $env:OLLAMA_OFFLOAD_MODEL = ''
            try {
                $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
                Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
                $script:Sent[-1].model | Should -Be 'qwen3:8b' -Because 'empty env must not ship as model; /api/ps must win'
            } finally {
                Remove-Item Env:OLLAMA_OFFLOAD_MODEL -ErrorAction SilentlyContinue
            }
        }

        It 'treats whitespace-only OLLAMA_OFFLOAD_MODEL as UNSET (tab/newline/NBSP/mixed)' {
            # DA-round4 (C) mirror of 02e04db: the whitespace category is not
            # one input but a family (spaces, tab, newline, NBSP, mixed). A
            # refactor to a hand-rolled strip on " `t`n" alone would leak
            # NBSP; parametrization catches it. .NET String.Trim() (the .Trim()
            # in the ladder) considers `[char]0x00A0` whitespace because
            # `Char.IsWhiteSpace` returns true for it — parity with Python's
            # `str.strip()` covering NBSP.
            foreach ($case in @(
                @{ Label = 'spaces';  Value = '   ' },
                @{ Label = 'tab';     Value = "`t" },
                @{ Label = 'newline'; Value = "`n" },
                @{ Label = 'nbsp';    Value = [string][char]0x00A0 },
                @{ Label = 'mixed';   Value = " `t `n " }
            )) {
                # Fresh module + mocks per sub-case so `$Script:DefaultModelSource`
                # state does not leak between cases (parity with Python's
                # `importlib.reload` per subtest).
                Remove-Module OllamaOffload -Force -ErrorAction SilentlyContinue
                Import-Module $script:ModulePath -Force
                Mock -ModuleName OllamaOffload Invoke-RestMethod {
                    if ($Uri -like '*/api/ps')   { return $script:Ps }
                    if ($Uri -like '*/api/show') { return $script:Show }
                    if ($Uri -like '*/api/tags') { return $script:Tags }
                    if ($Uri -like '*/api/chat') {
                        [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable))
                        return (New-ChatResponse -Content '{"ok":true}')
                    }
                    throw "no fixture for $Uri"
                }
                $script:Sent = [System.Collections.ArrayList]::new()
                $env:OLLAMA_OFFLOAD_MODEL = $case.Value
                try {
                    $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
                    Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
                    $script:Sent[-1].model | Should -Be 'qwen3:8b' `
                        -Because "whitespace-only env ($($case.Label)) must be UNSET after strip; /api/ps discovery must win — the wire body must NOT ship '$($case.Value)'"
                } finally {
                    Remove-Item Env:OLLAMA_OFFLOAD_MODEL -ErrorAction SilentlyContinue
                }
            }
        }

        It 'strips surrounding whitespace from a real OLLAMA_OFFLOAD_MODEL value' {
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') {
                    [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable))
                    return (New-ChatResponse -Content '{"ok":true}')
                }
                throw "no fixture for $Uri"
            }
            $script:Sent = [System.Collections.ArrayList]::new()
            $env:OLLAMA_OFFLOAD_MODEL = '  qwen3:8b  '
            try {
                $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
                Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
                $script:Sent[-1].model | Should -Be 'qwen3:8b' -Because 'the ladder must strip before shipping'
            } finally {
                Remove-Item Env:OLLAMA_OFFLOAD_MODEL -ErrorAction SilentlyContinue
            }
        }
    }

    Context 'F5 retry-success refreshes cached ContextTokens (872de70 A)' {

        It 'refreshes $Script:Discovered.ContextTokens when the F5 retry rewrites .Model' {
            # 872de70 mirror. On F5 retry-success in _InvokeOllamaCallOnce
            # (source == 'hint' after import, then _DiscoverModelForHost
            # succeeds against the resolved base_url), the block
            # rewrites $Script:Discovered.Model to the recovered model.
            # Because _Discover returns $Script:Discovered BY REFERENCE,
            # the F4 identity guard at :768 (`$effectiveModel -eq $d.Model`)
            # is TRUE by construction from call 2 onward -- so without a
            # sibling refresh of ContextTokens, the cached import-time
            # hint-era value ships forever. The test asserts that the
            # cache entry actually flipped to the /api/ps runtime value
            # (8192 from the Ps fixture) after the retry, not the
            # 262144 architectural value that /api/show would have
            # returned against the hint at import time.
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') {
                    [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable))
                    return (New-ChatResponse -Content '{"ok":true}')
                }
                throw "no fixture for $Uri"
            }
            $script:Sent = [System.Collections.ArrayList]::new()
            InModuleScope OllamaOffload {
                # Prime the module cache with the hint-era shape: Model
                # is the hardcoded hint, ContextTokens is the /api/show
                # architectural value (262144), source stamped 'hint' so
                # the F5 branch fires on the next call.
                $Script:Discovered = @{
                    Model = $Script:DefaultModelHint
                    ContextTokens = 262144
                    KeepAlive = '5m'
                    BaseUrl = 'http://test.invalid:11434'
                }
                $Script:DefaultModelSource = 'hint'
            }
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            # Call 1: F5 retry fires (source == 'hint'); _DiscoverModelForHost
            # resolves 'qwen3:8b' from /api/ps and stamps source='ps'.
            # The 872de70 mirror block ALSO refreshes .ContextTokens via
            # _DiscoverContextTokensForHost -> /api/ps runtime value 8192.
            $cached = InModuleScope OllamaOffload { $Script:Discovered }
            $cached.Model | Should -Be 'qwen3:8b' -Because 'F5 retry rewrote the cache Model'
            $cached.ContextTokens | Should -Be 8192 -Because 'the 872de70 sibling refresh must flip ContextTokens to the recovered runtime value, not leave the stale 262144'
            # Wire body on THIS call must already ship the fresh value.
            $script:Sent[-1].options.num_ctx | Should -Be 8192
            # Call 2: F5 branch skipped (source == 'ps' now); falls to the
            # else-branch that reads $d.Model, then the F4 guard at :768
            # sees $effectiveModel -eq $d.Model and reuses $d.ContextTokens
            # -- which the refresh above just corrected.
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            $script:Sent[-1].options.num_ctx | Should -Be 8192 -Because 'without the sibling refresh, call 2 would ship the stale 262144 via the F4 shortcut'
            $script:Sent[-1].options.num_ctx | Should -Not -Be 262144
        }
    }

    Context 'F5 refresh commits Model + ContextTokens atomically (02e04db B)' {

        It 'transitions Get-OllamaContextBytes across F5 refresh without a torn read' {
            # DA-round4 (D) mirror of 02e04db: Get-OllamaContextBytes reads
            # $Script:Discovered.ContextTokens lock-free. Under Fix B the F5
            # refresh commits .Model and .ContextTokens together under
            # $Script:F5RefreshLock, so successive reads observe (old, old)
            # or (new, new) — never a torn (new_model, old_tokens). Snapshot
            # `Get-OllamaContextBytes 0.5` at THREE points: pre-recovery
            # (hint-era 262144 primed on the cache), post-first-call (F5
            # flip to the recovered 8192 from /api/ps), post-second-call
            # (stable at 8192 — F5 branch skipped because source is no
            # longer 'hint'). Assert the transition happened once and later
            # reads are consistent.
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') {
                    [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable))
                    return (New-ChatResponse -Content '{"ok":true}')
                }
                throw "no fixture for $Uri"
            }
            $script:Sent = [System.Collections.ArrayList]::new()
            InModuleScope OllamaOffload {
                # Prime the cache with the hint-era shape so the F5 branch
                # fires on the next call — Model = hardcoded hint,
                # ContextTokens = the /api/show architectural 262144.
                $Script:Discovered = @{
                    Model = $Script:DefaultModelHint
                    ContextTokens = 262144
                    KeepAlive = '5m'
                    BaseUrl = 'http://test.invalid:11434'
                }
                $Script:DefaultModelSource = 'hint'
            }
            # Point 1: post-prime — reader sees the hint-era 262144.
            $cbImport = Get-OllamaContextBytes -Ratio 0.5
            $cbImport | Should -Be ([int](262144 * 0.5 * 4)) `
                -Because 'point 1: pre-recovery context_bytes reflects the primed hint-era ContextTokens (262144)'

            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            # Point 2: post-first-call — F5 flipped both under the lock;
            # /api/ps reports context_length = 8192 for qwen3:8b.
            $cbAfterFlip = Get-OllamaContextBytes -Ratio 0.5
            $cbAfterFlip | Should -Be ([int](8192 * 0.5 * 4)) `
                -Because 'point 2: post-first-call context_bytes reflects the F5-refreshed runtime value (8192) — the flip must be observable by the same reader that saw the hint-era value'
            $cbAfterFlip | Should -Not -Be $cbImport `
                -Because 'the transition must have happened — otherwise the refresh path is inert'

            # Second call: F5 branch skipped now (source != 'hint'); value
            # must remain stable across the F4 identity-guard shortcut.
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            $cbStable = Get-OllamaContextBytes -Ratio 0.5
            # Point 3: post-second-call — no further refresh, value stable.
            $cbStable | Should -Be $cbAfterFlip `
                -Because 'point 3: post-second-call context_bytes must match point 2 — no further F5 refresh runs because $Script:DefaultModelSource is no longer ''hint''; readers see a stable value across the lock boundary'
        }
    }

    Context 'environment namespace' {

        It 'never reads Ollama own variables' {
            # OLLAMA_HOST, OLLAMA_MODELS and OLLAMA_KEEP_ALIVE belong to Ollama
            # itself. A library that read them would change behaviour for anyone
            # who already has them set for the server or CLI.
            $source = Get-Content $script:ModulePath -Raw
            foreach ($stolen in @("'OLLAMA_HOST'", "'OLLAMA_MODELS'", "'OLLAMA_KEEP_ALIVE'")) {
                $source | Should -Not -BeLike "*$stolen*"
            }
        }
    }
}
