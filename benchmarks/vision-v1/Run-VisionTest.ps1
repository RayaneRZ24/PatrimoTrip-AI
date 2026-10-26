param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$RunLabel = "Qwen3.5-2B-Q4_K_M-ngl20-ctx4096",
    [ValidateRange(10, 3600)][int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
$endpoint = [Uri]$BaseUrl
if ($endpoint.Scheme -ne "http" -or $endpoint.Host -notin @("127.0.0.1", "localhost", "[::1]", "::1")) {
    throw "Use a loopback HTTP endpoint: this test must run locally."
}
$BaseUrl = $BaseUrl.TrimEnd("/")
$utf8 = New-Object System.Text.UTF8Encoding($false)
$manifest = Get-Content (Join-Path $PSScriptRoot "cases.json") -Raw -Encoding UTF8 | ConvertFrom-Json

# Validate all inputs before sending any request. Gold labels are never sent.
foreach ($case in $manifest.cases) {
    $imagePath = Join-Path $PSScriptRoot $case.image
    if ((Get-FileHash $imagePath -Algorithm SHA256).Hash -ine $case.sha256) {
        throw "Image checksum mismatch: $($case.case_id)"
    }
}
try {
    $health = Invoke-RestMethod "$BaseUrl/health" -TimeoutSec 15
    $models = Invoke-RestMethod "$BaseUrl/v1/models" -TimeoutSec 15
    $modelName = @($models.data)[0].id
    if (-not $modelName) { throw "No model returned by the server." }
} catch {
    throw "Local server not ready. Keep llama-server running, then retry. $($_.Exception.Message)"
}

$allowed = ($manifest.allowed_sites | ForEach-Object { "$($_.site_id) : $($_.name)" }) -join "`n"
$prompt = @"
Analyse uniquement l'image fournie. Les sites autorises sont :
$allowed

Ne force jamais une correspondance avec cette liste.
Si l'image ne permet pas de conclure, utilise status uncertain et site_id unknown.
Si tu reconnais un autre lieu, utilise status out_of_scope et site_id unknown.
Sinon, utilise status identified et l'identifiant du site reconnu.
Donne au maximum trois observations visibles. Separe-les de ton hypothese.
N'ajoute aucune date, periode historique ou information non visible.
Reponds uniquement avec un objet JSON, sans Markdown, avec exactement ces cles :
{"site_id":"identifiant ou unknown","status":"identified ou uncertain ou out_of_scope","observations":["observation visible"],"hypothesis":"hypothese breve"}
"@

$runId = (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + ([Guid]::NewGuid().ToString("N").Substring(0,6))
$output = Join-Path (Join-Path $PSScriptRoot "results") $runId
New-Item -ItemType Directory -Path $output -Force | Out-Null
$configuration = [ordered]@{
    run_id = $runId; label = $RunLabel; base_url = $BaseUrl
    utc_started = [DateTime]::UtcNow.ToString("o"); powershell = $PSVersionTable.PSVersion.ToString()
    case_manifest_sha256 = (Get-FileHash (Join-Path $PSScriptRoot "cases.json") -Algorithm SHA256).Hash
    temperature = 0; seed = 42; max_tokens = 256; cache_prompt = $false
    note = "Reference-image smoke test; not an independent evaluation. Server flags are not inferred from RunLabel."
}
[IO.File]::WriteAllText((Join-Path $output "configuration.json"), ($configuration | ConvertTo-Json -Depth 10), $utf8)
$rows = New-Object System.Collections.Generic.List[object]
$rawPath = Join-Path $output "responses.jsonl"
$gpuProcess = $null
if ($env:OS -eq "Windows_NT") {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $gpuProcess = Start-Process -FilePath $smi.Source -ArgumentList @(
                "--query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu",
                "--format=csv", "-l", "1"
            ) -NoNewWindow -RedirectStandardOutput (Join-Path $output "gpu.csv") `
                -RedirectStandardError (Join-Path $output "gpu-errors.txt") -PassThru
        } catch {
            Write-Warning "GPU sampling unavailable; inference tests will continue."
        }
    }
}

try {
foreach ($case in $manifest.cases) {
    Write-Host "Testing $($case.case_id) ..."
    $bytes = [IO.File]::ReadAllBytes((Join-Path $PSScriptRoot $case.image))
    $body = @{
        model = $modelName
        messages = @(@{ role = "user"; content = @(
            @{ type = "text"; text = $prompt },
            @{ type = "image_url"; image_url = @{ url = "data:image/jpeg;base64," + [Convert]::ToBase64String($bytes) } }
        ) })
        temperature = 0; seed = 42; max_tokens = 256; stream = $false; cache_prompt = $false
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 12 -Compress
    $reply = $null; $text = ""; $parsed = $null; $requestError = $null; $formatError = $null
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        $reply = Invoke-RestMethod "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $utf8.GetBytes($body) -TimeoutSec $TimeoutSeconds
        $text = [string]$reply.choices[0].message.content
    } catch {
        $requestError = $_.Exception.Message
    } finally {
        $watch.Stop()
    }
    $valid = $false
    if (-not $requestError) {
        try {
            $parsed = $text | ConvertFrom-Json -ErrorAction Stop
            $keys = @($parsed.PSObject.Properties.Name)
            $required = @("site_id", "status", "observations", "hypothesis")
            $sameKeys = $keys.Count -eq 4 -and @($required | Where-Object { $_ -notin $keys }).Count -eq 0
            $validId = $parsed.site_id -eq "unknown" -or $parsed.site_id -in @($manifest.allowed_sites.site_id)
            $validStatus = $parsed.status -in @("identified", "uncertain", "out_of_scope")
            $validObservations = $parsed.observations -is [Array] -and @($parsed.observations).Count -ge 1 -and @($parsed.observations).Count -le 3
            foreach ($observation in @($parsed.observations)) {
                if ($observation -isnot [string] -or -not $observation.Trim()) { $validObservations = $false }
            }
            $consistent = ($parsed.status -eq "identified" -and $parsed.site_id -ne "unknown") -or `
                ($parsed.status -in @("uncertain", "out_of_scope") -and $parsed.site_id -eq "unknown")
            $valid = $sameKeys -and $validId -and $validStatus -and $validObservations -and `
                $consistent -and $parsed.hypothesis -is [string]
            if (-not $valid) { $formatError = "Schema or field-consistency violation." }
        } catch {
            $formatError = "Not a strict JSON response: $($_.Exception.Message)"
        }
    }
    $identityCorrect = $null; $statusCorrect = $null; $predicted = ""; $status = ""
    if ($valid) {
        $predicted = $parsed.site_id; $status = $parsed.status
        $identityCorrect = $predicted -eq $case.expected_site_id
        $statusCorrect = $status -eq $case.expected_status
    }
    $record = [ordered]@{
        case_id = $case.case_id; category = $case.category; expected_site_id = $case.expected_site_id
        expected_status = $case.expected_status; seconds = [math]::Round($watch.Elapsed.TotalSeconds, 3)
        response_received = [bool]($null -ne $reply); strict_json_valid = [bool]$valid
        predicted_site_id = $predicted; predicted_status = $status
        identity_correct = $identityCorrect; status_correct = $statusCorrect
        request_error = $requestError; format_error = $formatError; response_text = $text
        finish_reason = $(if ($reply) { $reply.choices[0].finish_reason } else { $null })
        usage = $(if ($reply) { $reply.usage } else { $null })
        timings = $(if ($reply) { $reply.timings } else { $null })
        manual_description_review = "pending"; parsed_response = $parsed
    }
    [IO.File]::AppendAllText($rawPath, (($record | ConvertTo-Json -Depth 20 -Compress) + "`n"), $utf8)
    $rows.Add([pscustomobject]@{
        case_id = $case.case_id; category = $case.category; expected_site_id = $case.expected_site_id
        predicted_site_id = $predicted; status = $status; response_received = [bool]($null -ne $reply)
        strict_json_valid = [bool]$valid; identity_correct = $identityCorrect; status_correct = $statusCorrect
        seconds = $record.seconds; observations = $(if ($valid) { $parsed.observations -join " | " } else { "" })
        hypothesis = $(if ($valid) { $parsed.hypothesis } else { "" })
        request_error = $requestError; format_error = $formatError; manual_description_review = "pending"
    })
}
} finally {
    if ($gpuProcess -and -not $gpuProcess.HasExited) {
        Stop-Process -Id $gpuProcess.Id -ErrorAction SilentlyContinue
    }
}

$rows | Export-Csv (Join-Path $output "results.csv") -NoTypeInformation -Delimiter ";" -Encoding UTF8
$summary = [ordered]@{
    cases_planned = @($manifest.cases).Count; cases_attempted = $rows.Count
    responses_received = @($rows | Where-Object { $_.response_received }).Count
    valid_json = @($rows | Where-Object { $_.strict_json_valid }).Count
    standard_views_exact = @($rows | Where-Object { $_.category -eq "standard_view" -and $_.identity_correct -eq $true -and $_.status_correct -eq $true }).Count
    standard_views_total = @($rows | Where-Object { $_.category -eq "standard_view" }).Count
    distant_views_exact = @($rows | Where-Object { $_.category -eq "distant_view" -and $_.identity_correct -eq $true -and $_.status_correct -eq $true }).Count
    distant_views_total = @($rows | Where-Object { $_.category -eq "distant_view" }).Count
    out_of_scope_correct = @($rows | Where-Object { $_.category -eq "out_of_scope" -and $_.identity_correct -eq $true -and $_.status_correct -eq $true }).Count
    out_of_scope_total = @($rows | Where-Object { $_.category -eq "out_of_scope" }).Count
    note = "Invalid JSON and request failures are not counted as correct. Review distant-view abstentions and all descriptions manually. Small reference sample, not general model accuracy."
}
[IO.File]::WriteAllText((Join-Path $output "summary.json"), ($summary | ConvertTo-Json -Depth 10), $utf8)
$rows | Format-Table case_id, predicted_site_id, status, strict_json_valid, identity_correct, seconds -AutoSize
Write-Host "Results saved locally: $output"
Write-Host ($summary | ConvertTo-Json -Depth 10)
if ($summary.responses_received -eq 0) { throw "No model responses received: this run does not validate inference." }
