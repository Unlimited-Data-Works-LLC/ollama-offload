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

        It 'treats whitespace-only OLLAMA_OFFLOAD_MODEL as UNSET' {
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
            $env:OLLAMA_OFFLOAD_MODEL = '   '
            try {
                $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
                Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
                $script:Sent[-1].model | Should -Be 'qwen3:8b'
            } finally {
                Remove-Item Env:OLLAMA_OFFLOAD_MODEL -ErrorAction SilentlyContinue
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
