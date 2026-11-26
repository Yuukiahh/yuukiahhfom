<#
.SYNOPSIS
  Installs AutoClipper: a fully local, GPU-accelerated SupoClip (Whisper + Ollama) with the
  SupoClip MCP server for Claude Code, the AutoClipper toolkit and the AutoClipper skill.

.DESCRIPTION
  Safe to run again at any time: every step checks what is already there. Nothing here uses a
  paid API. Run it in a normal (non-admin) PowerShell; Windows asks for admin rights only for
  the installers that need them (WSL, Docker Desktop).

    powershell -ExecutionPolicy Bypass -File .\Install-AutoClipper.ps1

.PARAMETER InstallRoot   Where SupoClip and settings live (default: %USERPROFILE%\AutoClipper)
.PARAMETER OllamaModel   Local LLM for clip selection (default gpt-oss:20b, fits a 16 GB RTX 5070 Ti)
.PARAMETER WhisperModel  Whisper size: turbo (default, near large-v3 quality, ~6 GB VRAM), large-v3, medium
.PARAMETER SkipSelfTest  Do not run the end-to-end self test at the end
#>
[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path $env:USERPROFILE 'AutoClipper'),
    [string]$OllamaModel = 'gpt-oss:20b',
    [string]$WhisperModel = 'turbo',
    [string]$SupoClipCommit = '29ca74f9f38f7107bfca1939fd017ae7a0f147a8',
    [switch]$SkipSelfTest
)

# 'Continue': in Windows PowerShell 5.1, 'Stop' turns any stderr line of a native tool (git, docker,
# wsl) into an exception. Native results are checked through $LASTEXITCODE; cmdlets that matter
# use -ErrorAction Stop.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot   = Split-Path -Parent $PSScriptRoot          # ...\autoclipper
$SupoDir    = Join-Path $InstallRoot 'supoclip'
$LogFile    = Join-Path $InstallRoot 'install.log'
$ConfigFile = Join-Path $InstallRoot 'config.json'
$SecretFile = Join-Path $InstallRoot 'secrets.json'
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

function Log([string]$msg, [string]$color = 'Gray') {
    $line = '[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}
function Step([string]$msg) { Log ('==> ' + $msg) 'Cyan' }
function Ok([string]$msg)   { Log ('    OK  ' + $msg) 'Green' }
function Warn([string]$msg) { Log ('    !!  ' + $msg) 'Yellow' }
function Fail([string]$msg) { Log ('    XX  ' + $msg) 'Red'; throw $msg }

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user;$env:USERPROFILE\.local\bin"
}
function Have([string]$cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function New-Secret([int]$bytes = 32) {
    $b = New-Object byte[] $bytes
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    -join ($b | ForEach-Object { $_.ToString('x2') })
}

function Protect-File([string]$path) {
    # Only the current user (and SYSTEM) may read secrets.
    $acl = New-Object System.Security.AccessControl.FileSecurity
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($id in @("$env:USERDOMAIN\$env:USERNAME", 'NT AUTHORITY\SYSTEM')) {
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($id, 'FullControl', 'Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -Path $path -AclObject $acl -ErrorAction Stop
}

function Write-Utf8NoBom([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

function Winget-Install([string]$id, [string]$probe) {
    if ($probe -and (Have $probe)) { Ok "$id already installed"; return }
    $listed = winget list --id $id -e --accept-source-agreements 2>$null | Out-String
    if ($listed -match [regex]::Escape($id)) { Ok "$id already installed"; Refresh-Path; return }
    Log "    installing $id (winget verifies the publisher's installer hash) ..."
    winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements --disable-interactivity | Out-Null
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) { Fail "winget install $id failed ($LASTEXITCODE)" }
    Refresh-Path
    Ok "$id installed"
}

function Wait-Http([string]$url, [int]$seconds = 600) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10
            if ($r.StatusCode -lt 500) { return $true }
        } catch { }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Set-EnvLine([string[]]$lines, [string]$key, [string]$value) {
    $pattern = '^\s*#?\s*' + [regex]::Escape($key) + '='
    $found = $false
    $out = foreach ($l in $lines) {
        if (-not $found -and $l -match $pattern) { $found = $true; "$key=$value" } else { $l }
    }
    if (-not $found) { $out = @($out) + "$key=$value" }
    return , $out
}

Log "AutoClipper installer - log: $LogFile" 'White'

# --------------------------------------------------------------------------------------------
Step '1/12 System check (Windows, GPU, driver)'
$os = Get-CimInstance Win32_OperatingSystem
Log "    $($os.Caption) build $($os.BuildNumber), RAM $([math]::Round($os.TotalVisibleMemorySize/1MB)) GB"
if ([int]$os.BuildNumber -lt 19045) { Warn 'Windows 10 22H2 / Windows 11 recommended for Docker GPU support' }
if (-not (Have 'winget')) { Fail 'winget missing: install "App Installer" from the Microsoft Store, then rerun' }
$gpuOk = $false
if (Have 'nvidia-smi') {
    $gpu = (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) -join ' | '
    Ok "GPU: $gpu"
    $drv = 0
    try { $drv = [double](($gpu -split ',')[1].Trim().Split('.')[0]) } catch { }
    if ($gpu -match '50[0-9]0' -and $drv -lt 572) {
        Warn "RTX 50 series needs NVIDIA driver 572 or newer (found $drv): update it in the NVIDIA App, reboot, rerun"
    }
    $gpuOk = $true
} else {
    Warn 'nvidia-smi not found: install the current NVIDIA Game Ready/Studio driver; continuing CPU-only'
}

# --------------------------------------------------------------------------------------------
Step '2/12 Tools: Git, FFmpeg, uv, Ollama'
Winget-Install 'Git.Git' 'git'
Winget-Install 'Gyan.FFmpeg' 'ffmpeg'
Winget-Install 'astral-sh.uv' 'uv'
Winget-Install 'Ollama.Ollama' 'ollama'
Refresh-Path

# --------------------------------------------------------------------------------------------
Step '3/12 WSL2 + Docker Desktop'
$wsl = (wsl.exe --status 2>&1 | Out-String) -replace "`0", ''
if ($LASTEXITCODE -ne 0 -or $wsl -match 'not installed|nicht installiert') {
    Warn 'WSL2 is missing: installing (Windows asks for admin rights) ...'
    Start-Process -FilePath 'wsl.exe' -ArgumentList '--install', '--no-distribution' -Verb RunAs -Wait
    Log 'WSL2 installed. RESTART WINDOWS, then run this installer again.' 'Yellow'
    exit 10
}
Ok 'WSL2 present'
wsl.exe --update 2>$null | Out-Null
Winget-Install 'Docker.DockerDesktop' 'docker'
Refresh-Path
$dockerExe = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $dockerExe) { Start-Process $dockerExe }
    Log '    waiting for Docker Desktop (accept its terms in the window if it asks) ...'
    $deadline = (Get-Date).AddMinutes(8)
    do { Start-Sleep 5; docker info 2>$null | Out-Null } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) { Fail 'Docker Desktop did not start. Open it once, finish its setup, then rerun.' }
}
Ok 'Docker engine running'
# Docker Desktop: start with Windows so the clipping stack is always available.
$dockerSettings = Join-Path $env:APPDATA 'Docker\settings-store.json'
if (Test-Path $dockerSettings) {
    try {
        $ds = Get-Content $dockerSettings -Raw | ConvertFrom-Json
        if (-not $ds.AutoStart) { $ds | Add-Member -Force NoteProperty AutoStart $true; Write-Utf8NoBom $dockerSettings ($ds | ConvertTo-Json -Depth 20) }
    } catch { Warn 'could not enable Docker autostart (enable it in Docker Desktop > Settings > General)' }
}

# --------------------------------------------------------------------------------------------
Step '4/12 Ollama (local LLM) configuration + model'
# Long transcripts need a large context; flash attention + q8 KV cache keep it inside 16 GB VRAM.
foreach ($kv in @(@('OLLAMA_CONTEXT_LENGTH', '32768'), @('OLLAMA_FLASH_ATTENTION', '1'),
                   @('OLLAMA_KV_CACHE_TYPE', 'q8_0'), @('OLLAMA_KEEP_ALIVE', '10m'))) {
    if ([Environment]::GetEnvironmentVariable($kv[0], 'User') -ne $kv[1]) {
        [Environment]::SetEnvironmentVariable($kv[0], $kv[1], 'User')
        Set-Item -Path ("env:" + $kv[0]) -Value $kv[1]
        $restartOllama = $true
    }
}
if ($restartOllama) { Get-Process 'ollama*' -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2 }
if (-not (Wait-Http 'http://127.0.0.1:11434/api/version' 5)) {
    $app = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama app.exe'
    if (Test-Path $app) { Start-Process $app } else { Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden }
    if (-not (Wait-Http 'http://127.0.0.1:11434/api/version' 90)) { Fail 'Ollama does not answer on port 11434' }
}
Ok 'Ollama running'
$models = (ollama list) -join "`n"
if ($models -notmatch [regex]::Escape($OllamaModel)) {
    Log "    downloading $OllamaModel (about 14 GB, once) ..."
    ollama pull $OllamaModel
    if ($LASTEXITCODE -ne 0) { Fail "ollama pull $OllamaModel failed" }
}
Ok "model $OllamaModel ready"

# --------------------------------------------------------------------------------------------
Step '5/12 SupoClip source (pinned, reviewed commit)'
if (-not (Test-Path (Join-Path $SupoDir '.git'))) {
    git -c core.autocrlf=false clone --filter=blob:none https://github.com/FujiwaraChoki/supoclip.git $SupoDir
    if ($LASTEXITCODE -ne 0) { Fail 'git clone failed' }
}
Push-Location $SupoDir
git config core.autocrlf false
git fetch --quiet origin $SupoClipCommit 2>$null
git -c advice.detachedHead=false checkout --quiet --force $SupoClipCommit
if ($LASTEXITCODE -ne 0) { Fail "cannot check out SupoClip $SupoClipCommit" }
Pop-Location
Ok "SupoClip at $($SupoClipCommit.Substring(0,12))"

# --------------------------------------------------------------------------------------------
Step '6/12 SupoClip configuration (.env: Whisper + Ollama, GPU, no paid APIs)'
$secrets = @{}
if (Test-Path $SecretFile) { $secrets = @{}; (Get-Content $SecretFile -Raw | ConvertFrom-Json).psobject.Properties | ForEach-Object { $secrets[$_.Name] = $_.Value } }
foreach ($k in @('auth_secret', 'better_auth_secret', 'settings_key', 'redis_password', 'web_password')) {
    if (-not $secrets[$k]) { $secrets[$k] = New-Secret }
}
$envFile = Join-Path $SupoDir '.env'
$lines = if (Test-Path $envFile) { Get-Content $envFile -Encoding UTF8 } else { Get-Content (Join-Path $SupoDir '.env.example') -Encoding UTF8 }
$ollamaLlm = "ollama:$OllamaModel"
$settings = [ordered]@{
    'TRANSCRIPTION_PROVIDER'      = 'whisper'
    'WHISPER_MODEL_SIZE'          = $WhisperModel
    'BACKEND_CPU_ONLY'            = $(if ($gpuOk) { 'false' } else { 'true' })
    'BACKEND_PLATFORM'            = 'linux/amd64'
    'LLM'                         = $ollamaLlm
    'RERANK_LLM'                  = $ollamaLlm
    'AI_EDIT_MODEL'               = $ollamaLlm
    'EDIT_PLAN_LLM'               = $ollamaLlm
    'CLIP_EDGE_LLM'               = $ollamaLlm
    'OLLAMA_BASE_URL'             = 'http://host.docker.internal:11434/v1'
    'OPENAI_API_KEY'              = ''
    'OPENROUTER_API_KEY'          = ''
    'GOOGLE_API_KEY'              = ''
    'ANTHROPIC_API_KEY'           = ''
    'ASSEMBLY_AI_API_KEY'         = ''
    'APIFY_API_TOKEN'             = ''
    'SELF_HOST'                   = 'true'
    'ALLOW_UNSIGNED_BACKEND_AUTH' = 'false'
    'BACKEND_AUTH_SECRET'         = $secrets.auth_secret
    'BETTER_AUTH_SECRET'          = $secrets.better_auth_secret
    'APP_SETTINGS_ENCRYPTION_KEY' = $secrets.settings_key
    'REDIS_PASSWORD'              = $secrets.redis_password
    'FRONTEND_PORT'               = '3107'
    'MAX_CLIPS'                   = '15'
    'FAST_MODE_MAX_CLIPS'         = '15'
    'WORKFLOWS_PUBLIC'            = 'highlights,series,topic,recap,trailer'
    'COMPOSE_PROFILES'            = 'youtube-proxy'
    'YOUTUBE_POT_PROVIDER_URL'    = 'http://bgutil:4416'
    'POSTHOG_PROJECT_TOKEN'       = ''
    'NEXT_PUBLIC_POSTHOG_KEY'     = ''
}
if ($secrets.pexels_api_key) { $settings['PEXELS_API_KEY'] = $secrets.pexels_api_key }
foreach ($key in $settings.Keys) { $lines = Set-EnvLine $lines $key $settings[$key] }
Write-Utf8NoBom $envFile (($lines -join "`n") + "`n")
Protect-File $envFile
Copy-Item (Join-Path $PSScriptRoot 'templates\docker-compose.override.yml') (Join-Path $SupoDir 'docker-compose.override.yml') -Force -ErrorAction Stop
if (-not $gpuOk) {
    # No NVIDIA GPU: drop the GPU reservation so containers still start.
    $ov = Join-Path $SupoDir 'docker-compose.override.yml'
    Write-Utf8NoBom $ov ((Get-Content $ov -Raw) -replace '(?s)    deploy:.*?capabilities: \[gpu, compute, utility, video\]\r?\n', '')
}
Ok '.env written (secrets only readable by you)'

# --------------------------------------------------------------------------------------------
Step '7/12 Build and start SupoClip (first build takes 15-30 minutes)'
Push-Location $SupoDir
docker compose build 2>&1 | Tee-Object -FilePath (Join-Path $InstallRoot 'docker-build.log') | Select-String -Pattern 'ERROR|error:' | ForEach-Object { Warn $_.Line }
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail 'docker compose build failed (see docker-build.log)' }
docker compose up -d 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail 'docker compose up failed' }
Pop-Location
if (-not (Wait-Http 'http://127.0.0.1:8000/health' 600)) { Fail 'SupoClip backend not healthy (docker compose logs backend)' }
Ok 'backend http://localhost:8000'
if (-not (Wait-Http 'http://127.0.0.1:3107' 600)) { Warn 'web app on http://localhost:3107 not up yet' } else { Ok 'web app http://localhost:3107' }

# Ollama reachable from inside the containers?
$probe = docker exec supoclip-worker sh -c 'curl -s -m 10 http://host.docker.internal:11434/api/version' 2>$null
if (-not $probe) {
    Warn 'containers cannot reach Ollama on localhost: letting Ollama listen on all interfaces.'
    Warn 'If Windows Firewall asks about Ollama, allow PRIVATE networks only (never public).'
    [Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
    $env:OLLAMA_HOST = '0.0.0.0:11434'
    Get-Process 'ollama*' -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2
    Start-Process (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama app.exe')
    Wait-Http 'http://127.0.0.1:11434/api/version' 60 | Out-Null
    $probe = docker exec supoclip-worker sh -c 'curl -s -m 10 http://host.docker.internal:11434/api/version' 2>$null
    if (-not $probe) { Fail 'containers still cannot reach Ollama' }
}
Ok "containers reach Ollama ($probe)"

# GPU inside the worker: real CUDA op, catches 'no kernel image' on new GPUs (RTX 50 = sm_120).
if ($gpuOk) {
    $cuda = docker exec supoclip-worker .venv/bin/python -c "import torch;x=torch.ones(64,device='cuda');print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),float((x*2).sum()))" 2>&1
    if ($cuda -match '128\.0') { Ok "CUDA in worker: $cuda" } else { Warn "GPU not usable in worker, Whisper falls back to CPU: $cuda" }
}

# --------------------------------------------------------------------------------------------
Step '8/12 Local SupoClip account + dynamic edits'
$email = 'autoclipper@supoclip.local'
$userId = docker exec supoclip-postgres psql -U supoclip -d supoclip -tAc "SELECT id FROM users WHERE email='$email'"
if (-not $userId) {
    $body = @{ email = $email; password = $secrets.web_password; name = 'AutoClipper' } | ConvertTo-Json
    try {
        Invoke-RestMethod -Method Post -Uri 'http://localhost:3107/api/auth/sign-up/email' -Body $body `
            -ContentType 'application/json' -Headers @{ Origin = 'http://localhost:3107' } -ErrorAction Stop | Out-Null
    } catch { Fail "could not create the local account: $($_.Exception.Message)" }
    $userId = docker exec supoclip-postgres psql -U supoclip -d supoclip -tAc "SELECT id FROM users WHERE email='$email'"
}
$userId = "$userId".Trim()
if (-not $userId) { Fail 'local account not found after sign-up' }
# Self-hosted, single user: you are the admin; switch on dynamic edits (animated captions, SFX, zooms).
docker exec supoclip-postgres psql -U supoclip -d supoclip -c "UPDATE users SET is_admin=true WHERE id='$userId'; UPDATE feature_flags SET web_enabled=true WHERE key='dynamic_edits';" | Out-Null
docker compose -f (Join-Path $SupoDir 'docker-compose.yml') -f (Join-Path $SupoDir 'docker-compose.override.yml') --project-directory $SupoDir restart backend worker ai-worker 2>&1 | Out-Null
Wait-Http 'http://127.0.0.1:8000/health' 300 | Out-Null
$secrets['user_id'] = $userId
$secrets['web_login'] = $email
Write-Utf8NoBom $SecretFile ($secrets | ConvertTo-Json)
Protect-File $SecretFile
Ok "account $email (web login password in $SecretFile)"

# --------------------------------------------------------------------------------------------
Step '9/12 AutoClipper toolkit + SupoClip MCP server'
uv tool install --force --python 3.12 (Join-Path $RepoRoot 'toolkit') 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail 'uv tool install autoclipper failed' }
uv tool install --force --python 3.12 (Join-Path $SupoDir 'mcp') 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail 'uv tool install supoclip-mcp failed' }
Refresh-Path
$desktop = [Environment]::GetFolderPath('Desktop')     # follows OneDrive redirection
$outRoot = Join-Path $desktop 'AI Clips'
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
$config = [ordered]@{
    api_url = 'http://localhost:8000'
    output_root = $outRoot
    ollama_url = 'http://localhost:11434'
    ollama_model = $OllamaModel
    own_channels = @()
    min_virality = 55
    min_clips = 5
    max_clips = 15
    encoder = 'auto'
    pexels_configured = [bool]$secrets.pexels_api_key
}
if (Test-Path $ConfigFile) {
    # Keep what you changed by hand (own_channels, thresholds, task defaults).
    $old = Get-Content $ConfigFile -Raw | ConvertFrom-Json
    foreach ($p in $old.psobject.Properties) { if ($p.Name -notin @('api_url', 'ollama_model', 'pexels_configured')) { $config[$p.Name] = $p.Value } }
}
Write-Utf8NoBom $ConfigFile ($config | ConvertTo-Json -Depth 10)
Ok "toolkit installed, clips go to $outRoot"

# --------------------------------------------------------------------------------------------
Step '10/12 Claude Code: MCP server, skill, CLAUDE.md'
if (-not (Have 'claude')) {
    Warn 'Claude Code CLI not found: installing the official native build ...'
    Invoke-RestMethod https://claude.ai/install.ps1 | Invoke-Expression
    Refresh-Path
}
$mcpExe = (Get-Command supoclip-mcp -ErrorAction SilentlyContinue).Source
if (-not $mcpExe) { $mcpExe = Join-Path $env:USERPROFILE '.local\bin\supoclip-mcp.exe' }
if (Have 'claude') {
    claude mcp remove supoclip --scope user 2>$null | Out-Null
    claude mcp add supoclip --scope user `
        -e SUPOCLIP_API_URL=http://localhost:8000 `
        -e "SUPOCLIP_USER_ID=$userId" `
        -e "SUPOCLIP_AUTH_SECRET=$($secrets.auth_secret)" `
        -e "SUPOCLIP_DOWNLOAD_DIR=$(Join-Path $InstallRoot 'mcp-downloads')" `
        -e SUPOCLIP_TIMEOUT=120 `
        -- $mcpExe | Out-Null
    if ($LASTEXITCODE -eq 0) { Ok 'MCP server "supoclip" registered for Claude Code (user scope)' } else { Warn 'claude mcp add failed' }
} else { Warn 'Claude Code not available: MCP not registered' }

$claudeHome = Join-Path $env:USERPROFILE '.claude'
$skillDest = Join-Path $claudeHome 'skills\autoclipper'
New-Item -ItemType Directory -Force -Path $skillDest | Out-Null
Copy-Item -Recurse -Force (Join-Path $RepoRoot 'skill\autoclipper\*') $skillDest -ErrorAction Stop
Ok "skill installed: $skillDest"
$binDir = Join-Path $InstallRoot 'bin'
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
Copy-Item -Force (Join-Path $PSScriptRoot '*.ps1') $binDir -ErrorAction Stop
Copy-Item -Recurse -Force (Join-Path $PSScriptRoot 'templates') $binDir -ErrorAction Stop
Ok "helper scripts in $binDir (Start-AutoClipper.ps1, Test-AutoClipper.ps1)"

$claudeMd = Join-Path $claudeHome 'CLAUDE.md'
$block = Get-Content (Join-Path $RepoRoot 'CLAUDE.md') -Raw -Encoding UTF8
$block = $block.Replace('{{OUTPUT_ROOT}}', $outRoot).Replace('{{INSTALL_ROOT}}', $InstallRoot)
$wrapped = "<!-- AUTOCLIPPER:BEGIN (managed by Install-AutoClipper.ps1) -->`n$block`n<!-- AUTOCLIPPER:END -->"
$current = if (Test-Path $claudeMd) { Get-Content $claudeMd -Raw -Encoding UTF8 } else { '' }
if ($current -match '(?s)<!-- AUTOCLIPPER:BEGIN.*?<!-- AUTOCLIPPER:END -->') {
    $current = [regex]::Replace($current, '(?s)<!-- AUTOCLIPPER:BEGIN.*?<!-- AUTOCLIPPER:END -->', [System.Text.RegularExpressions.MatchEvaluator] { param($m) $wrapped })
} else {
    $current = ($current.TrimEnd() + "`n`n" + $wrapped).TrimStart()
}
Write-Utf8NoBom $claudeMd ($current + "`n")
Ok "CLAUDE.md updated: $claudeMd"

# Let the AutoClipper commands run without a permission prompt each time.
$settingsPath = Join-Path $claudeHome 'settings.json'
$cs = $null
if (Test-Path $settingsPath) { $cs = Get-Content $settingsPath -Raw | ConvertFrom-Json }
if (-not $cs) { $cs = [pscustomobject]@{} }
if (-not $cs.permissions) { $cs | Add-Member -Force NoteProperty permissions ([pscustomobject]@{ allow = @() }) }
if (-not $cs.permissions.allow) { $cs.permissions | Add-Member -Force NoteProperty allow @() }
$allow = [System.Collections.ArrayList]@($cs.permissions.allow)
foreach ($rule in @('Bash(autoclipper:*)', 'mcp__supoclip')) {
    if (-not $allow.Contains($rule)) { [void]$allow.Add($rule) }
}
$cs.permissions.allow = $allow.ToArray()
Write-Utf8NoBom $settingsPath ($cs | ConvertTo-Json -Depth 20)
Ok 'Claude Code permissions for AutoClipper set'

# --------------------------------------------------------------------------------------------
Step '11/12 Health check'
autoclipper doctor
if ($LASTEXITCODE -ne 0) { Warn 'doctor reported a problem (see above)' } else { Ok 'all components answer' }

# --------------------------------------------------------------------------------------------
if (-not $SkipSelfTest) {
    Step '12/12 End-to-end self test (generated speech video, legally yours)'
    & (Join-Path $PSScriptRoot 'Test-AutoClipper.ps1')
}
Log ''
Log 'AutoClipper is installed. In Claude Code just paste a YouTube link.' 'Green'
Log "Clips: $outRoot   Web app: http://localhost:3107   Log: $LogFile" 'Green'
