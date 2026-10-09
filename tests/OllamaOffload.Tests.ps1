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

        It 'keep_alive auto: does NOT send the field when the model has no Forever pin (twin parity)' {
            # The shared fixture's /api/ps entry carries no expires_at: not a Forever pin.
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            $script:Sent[-1].Keys | Should -Not -Contain 'keep_alive'
        }
    }

    Context 'keep_alive auto conforms to the loaded model' {

        BeforeEach {
            $script:Sent = [System.Collections.ArrayList]::new()
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:PsNow }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') {
                    [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable))
                    return (New-ChatResponse -Content '{"ok":true}')
                }
                throw "no fixture for $Uri"
            }
            $script:Schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
        }

        It 're-asserts a Forever pin as the INTEGER -1' {
            $script:PsNow = [pscustomobject]@{ models = @([pscustomobject]@{ name = 'qwen3:8b'; context_length = 8192; expires_at = '2319-01-18T22:22:11.2222337Z' }) }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema | Out-Null
            $script:Sent[-1]['keep_alive'] | Should -Be -1
            $script:Sent[-1]['keep_alive'] | Should -BeOfType [long]
        }

        It 'sends no field for a finite TTL' {
            $script:PsNow = [pscustomobject]@{ models = @([pscustomobject]@{ name = 'qwen3:8b'; context_length = 8192; expires_at = '2026-10-09T12:05:00Z' }) }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema | Out-Null
            $script:Sent[-1].Keys | Should -Not -Contain 'keep_alive'
        }

        It 're-asserts Forever from a DESERIALIZED /api/ps (expires_at arrives as [DateTime])' {
            # Invoke-RestMethod/ConvertFrom-Json turn an ISO string into a [DateTime]; reading its first four
            # characters gave culture text ("02/1"), never the year. Feed real JSON text, not a hand-built string.
            $script:PsNow = '{"models":[{"name":"qwen3:8b","context_length":8192,"expires_at":"2318-02-10T14:42:57.1234567Z"}]}' | ConvertFrom-Json
            $script:PsNow.models[0].expires_at | Should -BeOfType [datetime]
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema | Out-Null
            $script:Sent[-1]['keep_alive'] | Should -Be -1
        }

        It 'keeps a confirmed Forever pin through a /api/ps probe miss' {
            $script:PsNow = '{"models":[{"name":"qwen3:8b","context_length":8192,"expires_at":"2318-02-10T14:42:57Z"}]}' | ConvertFrom-Json
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema | Out-Null
            $script:Sent[-1]['keep_alive'] | Should -Be -1
            InModuleScope OllamaOffload { $Script:Discovered = $Script:Discovered }   # discovery stays cached
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { throw 'timed out' }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable)); return (New-ChatResponse -Content '{"ok":true}') }
                throw "no fixture for $Uri"
            }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema | Out-Null
            $script:Sent[-1]['keep_alive'] | Should -Be -1
        }

        It 'matches an untagged model to its :latest tag' {
            $script:PsNow = '{"models":[{"name":"qwen3:latest","context_length":8192,"expires_at":"2318-02-10T14:42:57Z"}]}' | ConvertFrom-Json
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema -Model 'qwen3' | Out-Null
            $script:Sent[-1]['keep_alive'] | Should -Be -1
        }

        It 'sends env OLLAMA_OFFLOAD_KEEP_ALIVE=-1 as the INTEGER (the string is a 400)' {
            $env:OLLAMA_OFFLOAD_KEEP_ALIVE = '-1'
            try {
                $script:PsNow = $script:Ps
                Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema | Out-Null
                $script:Sent[-1]['keep_alive'] | Should -Be -1
                $script:Sent[-1]['keep_alive'] | Should -BeOfType [long]
            } finally { Remove-Item Env:OLLAMA_OFFLOAD_KEEP_ALIVE -ErrorAction SilentlyContinue }
        }

        It 'honours an explicit keep_alive' {
            $script:PsNow = [pscustomobject]@{ models = @([pscustomobject]@{ name = 'qwen3:8b'; context_length = 8192; expires_at = '2319-01-18T22:22:11Z' }) }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema -KeepAlive '30m' | Out-Null
            $script:Sent[-1]['keep_alive'] | Should -Be '30m'
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

    Context 'public status (twin of Python status() / clear_cooldown())' {

        It 'reports the discovered model, the num_ctx it would send, and a confirmed window' {
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                throw "no fixture for $Uri"
            }
            $st = Get-OllamaStatus
            $st.model | Should -Be 'qwen3:8b'
            $st.model_source | Should -Be 'cfg'   # the shipped config pins qwen3:8b; a pin outranks /api/ps
            $st.context_tokens | Should -Be 8192
            $st.context_confirmed | Should -BeTrue
            $st.cooldown_s | Should -Be 0
            $st.host | Should -Be (Get-OllamaActiveHost)
            @($st.PSObject.Properties.Name | Sort-Object) | Should -Be @('base_url','context_confirmed','context_tokens','cooldown_s','host','model','model_source')
        }

        It 'says unconfirmed when no window can be read' {
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return [pscustomobject]@{ models = @([pscustomobject]@{ name = 'qwen3:8b' }) } }
                if ($Uri -like '*/api/show') { throw 'busy' }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                throw "no fixture for $Uri"
            }
            (Get-OllamaStatus).context_confirmed | Should -BeFalse
        }

        It 'reports and clears a cooldown' {
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                throw "no fixture for $Uri"
            }
            InModuleScope OllamaOffload { _SetCooldownFor -HostName (Get-OllamaActiveHost) -Seconds 300 }
            (Get-OllamaStatus).cooldown_s | Should -BeGreaterThan 0
            @(Clear-OllamaCooldown).Count | Should -BeGreaterThan 0
            (Get-OllamaStatus).cooldown_s | Should -Be 0
        }
    }

    Context 'budget refusal (-InputRatio, twin of Python input_ratio)' {

        BeforeEach {
            $script:Sent = [System.Collections.ArrayList]::new()
            $script:VersionDown = $false
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/version') { if ($script:VersionDown) { throw 'connection refused' } return @{ version = '0.12.0' } }
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable)); return (New-ChatResponse -Content '{"ok":true}') }
                throw "no fixture for $Uri"
            }
            $script:Schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
        }

        It 'sends a prompt within the budget whole' {
            $p = 'a' * 1000   # 8192 x 0.4 x 4 = 13107
            Invoke-OllamaCall -TaskPrompt $p -Schema $script:Schema -InputRatio 0.4 | Out-Null
            $script:Sent[-1].messages[-1].content | Should -Be $p
        }

        It 'refuses an oversize prompt with its sizes, sends nothing, arms no cooldown' {
            $err = { Invoke-OllamaCall -TaskPrompt ('b' * 20000) -Schema $script:Schema -InputRatio 0.4 } | Should -Throw -PassThru
            $e = $err.Exception
            $e.GetType().Name | Should -Be 'OllamaPromptTooLarge'
            $e.GetType().BaseType.Name | Should -Be 'OllamaCallError'
            $e.Chars | Should -Be 20000
            $e.Budget | Should -Be 13107
            $e.ContextTokens | Should -Be 8192
            $script:Sent.Count | Should -Be 0
            (InModuleScope OllamaOffload { _CooldownRemainingFor (Get-OllamaActiveHost) }) | Should -Be 0
        }

        It 'counts the system prompt' {
            { Invoke-OllamaCall -TaskPrompt ('c' * 10000) -System ('s' * 4000) -InputRatio 0.4 } | Should -Throw
            $script:Sent.Count | Should -Be 0
        }

        It 'reports a down host as OllamaUnavailable, not too large, and arms the cooldown' {
            $script:VersionDown = $true
            $err = { Invoke-OllamaCall -TaskPrompt ('d' * 20000) -Schema $script:Schema -InputRatio 0.4 } | Should -Throw -PassThru
            $err.Exception.GetType().Name | Should -Be 'OllamaUnavailable'
            (InModuleScope OllamaOffload { _CooldownRemainingFor (Get-OllamaActiveHost) }) | Should -BeGreaterThan 0
        }

        It 'does no check without -InputRatio' {
            Invoke-OllamaCall -TaskPrompt ('e' * 20000) -Schema $script:Schema | Out-Null
            $script:Sent.Count | Should -Be 1
        }
    }

    Context 'a host that is down during discovery (twin of Python TestDeadHostAtImport)' {

        BeforeEach {
            Mock -ModuleName OllamaOffload Invoke-RestMethod { throw 'connection refused' }
            # No model pin anywhere, so discovery is what fails (the shipped config pins one).
            InModuleScope OllamaOffload {
                $cfg = Get-OllamaConfig
                $cfg.Remove('model')
                if ($cfg.ContainsKey('hosts')) { foreach ($h in @($cfg['hosts'].Values)) { if ($h -is [System.Collections.IDictionary]) { $h.Remove('model') } } }
                $Script:Discovered = $null
            }
        }

        It 'is OllamaUnavailable and benches the host' {
            $err = { Invoke-OllamaCall -TaskPrompt 'go' } | Should -Throw -PassThru
            $err.Exception.GetType().Name | Should -Be 'OllamaUnavailable'
            (InModuleScope OllamaOffload { _CooldownRemainingFor (Get-OllamaActiveHost) }) | Should -BeGreaterThan 0
        }

        It 'does not probe a benched host again' {
            InModuleScope OllamaOffload { _SetCooldownFor -HostName (Get-OllamaActiveHost) -Seconds 300 }
            $err = { Invoke-OllamaCall -TaskPrompt 'go' } | Should -Throw -PassThru
            $err.Exception.GetType().Name | Should -Be 'OllamaUnavailable'
            Should -Invoke -ModuleName OllamaOffload Invoke-RestMethod -Times 0 -Exactly
        }
    }

    Context 'PowerShell walks the same model ladder as Python, on the wire and in status' {

        BeforeEach {
            $script:Sent = [System.Collections.ArrayList]::new()
            $script:PsNow = '{"models":[{"name":"a:1b","context_length":4096}]}' | ConvertFrom-Json
            $script:TagsNow = '{"models":[{"name":"a:1b"}]}' | ConvertFrom-Json
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/version') { return @{ version = '0.12.0' } }
                if ($Uri -like '*/api/ps')   { return $script:PsNow }
                if ($Uri -like '*/api/show') { return [pscustomobject]@{ parameters = "num_ctx 16384" } }
                if ($Uri -like '*/api/tags') { return $script:TagsNow }
                if ($Uri -like '*/api/chat') { [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable)); return (New-ChatResponse -Content '{"ok":true}') }
                throw "no fixture for $Uri"
            }
            # A hosts map whose active host PINS b:1b while /api/ps has a:1b loaded (no raw-URL override).
            Remove-Item Env:OLLAMA_OFFLOAD_URL -ErrorAction SilentlyContinue
            InModuleScope OllamaOffload {
                $cfg = Get-OllamaConfig
                $cfg.Remove('model')
                $cfg['default_host'] = 'f'
                $cfg['hosts'] = @{ f = @{ url = 'http://test.invalid:11434/api/chat'; model = 'b:1b' } }
                $Script:Discovered = $null
                $Script:ActiveHost = $null   # resolved at import under the raw-URL env; re-resolve under this config
            }
            $script:Schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
        }

        It 'sends the host pin, not the loaded model (Python parity)' {
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $script:Schema | Out-Null
            $script:Sent[-1]['model'] | Should -Be 'b:1b'
        }

        It 'status reports the host pin as cfg, matching the wire' {
            $st = Get-OllamaStatus
            $st.model | Should -Be 'b:1b'
            $st.model_source | Should -Be 'cfg'
        }

        It 'status on the hint path reports what the call would rediscover, not the hint' {
            InModuleScope OllamaOffload {
                (Get-OllamaConfig)['hosts']['f'].Remove('model')
                $Script:Discovered = @{ Model = $Script:DefaultModelHint; ContextTokens = 8192; KeepAlive = '5m'; BaseUrl = 'http://test.invalid:11434' }
                $Script:DefaultModelSource = 'hint'
                # A pinned host never reaches the startup-miss recovery path (the pin outranks discovery): drop the pin.
                (Get-OllamaConfig).Remove('model'); $Script:ActiveHost = $null
            }
            $st = Get-OllamaStatus
            $st.model | Should -Be 'a:1b'
            $st.model_source | Should -Not -Be 'hint'
        }
    }

    Context 'Get-OllamaStatus has no side effects' {

        It 'status on the hint path, then a call: the wire carries what status reported' {
            $script:Sent = [System.Collections.ArrayList]::new()
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/version') { return @{ version = '0.12.0' } }
                if ($Uri -like '*/api/ps')   { return ('{"models":[]}' | ConvertFrom-Json) }
                if ($Uri -like '*/api/tags') { return ('{"models":[{"name":"a:1b"}]}' | ConvertFrom-Json) }
                if ($Uri -like '*/api/show') { return [pscustomobject]@{ parameters = "num_ctx 2048" } }
                if ($Uri -like '*/api/chat') { [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable)); return (New-ChatResponse -Content '{"ok":true}') }
                throw "no fixture for $Uri"
            }
            InModuleScope OllamaOffload {
                (Get-OllamaConfig).Remove('model'); $Script:ActiveHost = $null
                $Script:Discovered = @{ Model = $Script:DefaultModelHint; ContextTokens = 8192; KeepAlive = '5m'; BaseUrl = 'http://test.invalid:11434' }
                $Script:DefaultModelSource = 'hint'
            }
            $st = Get-OllamaStatus
            $st.model | Should -Be 'a:1b'
            (InModuleScope OllamaOffload { $Script:DefaultModelSource }) | Should -Be 'hint' -Because 'status must not change module state'
            Invoke-OllamaCall -TaskPrompt 'go' -Schema @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') } | Out-Null
            $script:Sent[-1]['model'] | Should -Be $st.model
        }
    }

    Context 'memo repair and tag-aware context probe' {

        It 'repairs a corrupt memo on the next successful probe and keeps the pin through a later miss' {
            $script:Sent = [System.Collections.ArrayList]::new()
            $script:PsNow = '{"models":[{"name":"qwen3:8b","context_length":8192,"expires_at":"2318-02-10T14:42:57Z"}]}' | ConvertFrom-Json
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:PsNow }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { [void]$script:Sent.Add(($Body | ConvertFrom-Json -AsHashtable)); return (New-ChatResponse -Content '{"ok":true}') }
                throw "no fixture for $Uri"
            }
            $memo = InModuleScope OllamaOffload { _KeepAliveMemoPath -BaseUrl 'http://test.invalid:11434' }
            Set-Content -Path $memo -Value '{bad json' -Encoding utf8
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null
            (Get-Content -Raw $memo | ConvertFrom-Json -AsHashtable)['qwen3:8b'] | Should -BeTrue
        }

        It 'matches an untagged name to /api/ps for the RUNTIME window' {
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return ('{"models":[{"name":"qwen3:latest","context_length":12345}]}' | ConvertFrom-Json) }
                if ($Uri -like '*/api/show') { return [pscustomobject]@{ parameters = "num_ctx 16384" } }
                throw "no fixture for $Uri"
            }
            (InModuleScope OllamaOffload { _ProbeContextTokensForHostStrict -Model 'qwen3' -BaseUrl 'http://test.invalid:11434' }) | Should -Be 12345
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

        It 'raises OllamaCallError when a schema reply lacks a required key (twin parity with Python)' {
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { return (New-ChatResponse -Content '{"other":1}') }
                throw "no fixture for $Uri"
            }
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            $err = { Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema } | Should -Throw -PassThru
            $err.Exception.GetType().Name | Should -Be 'OllamaCallError'
            $remaining = InModuleScope OllamaOffload { _CooldownRemainingFor (Get-OllamaActiveHost) }
            $remaining | Should -Be 0
        }

        It 'raises OllamaCallError when a schema reply is JSON but not an object' {
            foreach ($reply in @('42', '["ok"]', '[{"ok":true}]')) {   # the last: a one-element array must stay an array
                $script:Reply = $reply
                Mock -ModuleName OllamaOffload Invoke-RestMethod {
                    if ($Uri -like '*/api/ps')   { return $script:Ps }
                    if ($Uri -like '*/api/show') { return $script:Show }
                    if ($Uri -like '*/api/tags') { return $script:Tags }
                    if ($Uri -like '*/api/chat') { return (New-ChatResponse -Content $script:Reply) }
                    throw "no fixture for $Uri"
                }
                $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
                $err = { Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema } | Should -Throw -PassThru
            $err.Exception.GetType().Name | Should -Be 'OllamaCallError'
            }
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
                # A pinned host never reaches the startup-miss recovery path (the pin outranks discovery): drop the pin.
                (Get-OllamaConfig).Remove('model'); $Script:ActiveHost = $null
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
                # A pinned host never reaches the startup-miss recovery path (the pin outranks discovery): drop the pin.
                (Get-OllamaConfig).Remove('model'); $Script:ActiveHost = $null
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

    Context 'F5 strict probe runs UNDER the refresh lock (1fbf67c DA-r5)' {

        It 'holds $Script:F5RefreshLock while _ProbeContextTokensForHostStrict runs' {
            # DA-round5 (composition) mirror of 1fbf67c. Fix-A (strict-$null
            # skips the ctx write) and Fix-B (commit the pair under
            # $Script:F5RefreshLock) COMPOSE into a cross-model race unless
            # the strict probe runs INSIDE the lock. Repro under a probe-
            # outside-lock shape: writer B probes ctx=V for model B strict
            # → V, takes lock, writes (Model=B, ContextTokens=V), releases.
            # Writer A had probed for model A concurrently, /api/show
            # errored → strict returns $null; A takes lock, writes Model=A
            # but SKIPS ctx per Fix A. Final pair: (Model=A, ContextTokens
            # =V-for-B) — a cross-model pair, exactly what the lock was
            # supposed to prevent.
            #
            # The prior atomic-refresh test at :303 is an ISOLATION test —
            # no concurrency, asserts only single-value equality across a
            # sequential three-point read. Move the probe back OUTSIDE the
            # lock and prior tests still pass. That makes them a rubber-
            # stamp for the atomic-pair claim; this test is the anti-
            # rubber-stamp.
            #
            # PowerShell threading semantics don't allow a faithful
            # concurrent-writer test in-process — Start-Job forks a new
            # runspace with a FRESH copy of module state, so a second job
            # cannot observe THIS runspace's $Script:F5RefreshLock at all.
            # Fall back to a single-thread test that mocks
            # `_ProbeContextTokensForHostStrict` to record
            # $Script:F5RefreshLock.CurrentCount inside the probe body.
            # SemaphoreSlim.CurrentCount is 0 when the lock is HELD (all
            # slots taken) and 1 when free. Under 1fbf67c the recorded
            # value must be 0 on every probe entry from the F5 branch.
            # Under the prior af69a5d shape (probe BEFORE the .Wait()) the
            # recorded value would be 1.
            $script:ProbeLockCounts = [System.Collections.ArrayList]::new()
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { return (New-ChatResponse -Content '{"ok":true}') }
                throw "no fixture for $Uri"
            }
            Mock -ModuleName OllamaOffload _ProbeContextTokensForHostStrict {
                param($Model, $BaseUrl)
                # SemaphoreSlim.CurrentCount == 0 <=> lock is HELD.
                [void]$script:ProbeLockCounts.Add([int]$Script:F5RefreshLock.CurrentCount)
                return 8192
            }
            InModuleScope OllamaOffload {
                # Prime the hint-era shape so the F5 branch fires.
                $Script:Discovered = @{
                    Model = $Script:DefaultModelHint
                    ContextTokens = 262144
                    KeepAlive = '5m'
                    BaseUrl = 'http://test.invalid:11434'
                }
                $Script:DefaultModelSource = 'hint'
                # A pinned host never reaches the startup-miss recovery path (the pin outranks discovery): drop the pin.
                (Get-OllamaConfig).Remove('model'); $Script:ActiveHost = $null
            }
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null

            $script:ProbeLockCounts.Count | Should -BeGreaterOrEqual 1 `
                -Because 'F5 branch must have exercised the strict probe — otherwise the test is inert'
            foreach ($count in $script:ProbeLockCounts) {
                $count | Should -Be 0 `
                    -Because 'the strict probe must run INSIDE $Script:F5RefreshLock (CurrentCount == 0 means the semaphore is HELD); a value of 1 means the probe ran BEFORE the .Wait(), reopening the DA-r5 cross-model composition race'
            }
        }
    }

    Context 'F5 writer pair coherence under stubborn strict-None (f605783 DA-r6)' {

        It 'skips BOTH .Model AND .ContextTokens writes when strict probe returns $null' {
            # DA-round6 (residual) mirror of f605783. Even with the probe
            # INSIDE the lock (1fbf67c / DA-r5), a stubborn transient
            # (persistent /api/show 500, model-reload window) can still
            # return $null to a writer holding the lock. The prior shape
            # wrote .Model unconditionally and skipped .ContextTokens on
            # $null, so a serialized (B, then A) run of two writers ends
            # with (.Model=A, .ContextTokens=V-belonging-to-B) — a cross-
            # model pair the r5 lock was supposed to prevent.
            #
            # Fix: atomic all-or-nothing. On strict-$null, SKIP BOTH
            # writes and preserve whatever coherent pair
            # $Script:Discovered already held. A forfeited advance is
            # cheaper than a corrupted global; the next call re-attempts.
            #
            # PowerShell threading semantics: Start-Job forks a fresh
            # runspace with a NEW copy of module state, so a second job
            # cannot observe THIS runspace's $Script:Discovered at all —
            # a faithful two-writer concurrency test is not expressible
            # in-process (documented on the r5 test at :387). Fall back
            # to a deterministic single-thread mock: pre-set
            # $Script:Discovered to a coherent hint pair, force the F5
            # branch, mock the strict probe to return $null, then assert
            # BOTH fields are unchanged after Invoke-OllamaCall returns.
            # That locks the skip-BOTH behavior against a regression to
            # "advance .Model, skip only .ContextTokens".
            $HINT_CTX = 262144
            Mock -ModuleName OllamaOffload Invoke-RestMethod {
                if ($Uri -like '*/api/ps')   { return $script:Ps }
                if ($Uri -like '*/api/show') { return $script:Show }
                if ($Uri -like '*/api/tags') { return $script:Tags }
                if ($Uri -like '*/api/chat') { return (New-ChatResponse -Content '{"ok":true}') }
                throw "no fixture for $Uri"
            }
            # _ProbeContextTokensForHostStrict: the stubborn transient —
            # always $null. _DiscoverModelForHost runs naturally against
            # the mocked /api/ps (returns 'qwen3:8b', flips source to
            # 'ps'), so .Model would ADVANCE from the seeded hint value
            # to 'qwen3:8b' under the buggy shape while .ContextTokens
            # stays at $HINT_CTX (a cross-model pair). Under the fix,
            # BOTH writes are skipped and both fields remain at the
            # pre-F5 hint values.
            Mock -ModuleName OllamaOffload _ProbeContextTokensForHostStrict {
                param($Model, $BaseUrl)
                return $null
            }
            InModuleScope OllamaOffload {
                $Script:Discovered = @{
                    Model = $Script:DefaultModelHint
                    ContextTokens = 262144
                    KeepAlive = '5m'
                    BaseUrl = 'http://test.invalid:11434'
                }
                $Script:DefaultModelSource = 'hint'
                # A pinned host never reaches the startup-miss recovery path (the pin outranks discovery): drop the pin.
                (Get-OllamaConfig).Remove('model'); $Script:ActiveHost = $null
            }
            $HINT_MODEL = InModuleScope OllamaOffload { $Script:DefaultModelHint }
            $schema = @{ type = 'object'; properties = @{ ok = @{ type = 'boolean' } }; required = @('ok') }
            Invoke-OllamaCall -TaskPrompt 'go' -Schema $schema | Out-Null

            $final = InModuleScope OllamaOffload { $Script:Discovered }
            $final.Model | Should -Be $HINT_MODEL `
                -Because 'on strict-$null the F5 refresh must SKIP the .Model write — advancing .Model to the naturally-discovered value (qwen3:8b) while .ContextTokens stays at the hint-era 262144 is precisely the DA-round6 cross-model pair bug'
            $final.ContextTokens | Should -Be $HINT_CTX `
                -Because 'on strict-$null the F5 refresh must SKIP the .ContextTokens write too — atomic all-or-nothing, either both writes commit or neither does'
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
