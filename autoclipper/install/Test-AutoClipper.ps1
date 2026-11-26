<#
.SYNOPSIS
  Real end-to-end test. Generates a ~3 minute German talk with the Windows speech engine
  (so the test video is yours to use), runs the whole pipeline on it and checks the export.

.PARAMETER YouTubeUrl  Optional: additionally clip a YouTube video. Only Creative Commons videos
                       (or your own channels from config.json) pass without asking.
#>
param(
    [string]$InstallRoot = (Join-Path $env:USERPROFILE 'AutoClipper'),
    [string]$YouTubeUrl = ''
)
$ErrorActionPreference = 'Continue'   # see Install-AutoClipper.ps1: native stderr must not throw in PS 5.1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dir = Join-Path $InstallRoot 'selftest'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$wav = Join-Path $dir 'selftest.wav'
$mp4 = Join-Path $dir 'AutoClipper Selbsttest.mp4'

$script = @'
Wusstest du, dass ein Oktopus drei Herzen hat? Zwei davon pumpen Blut durch die Kiemen, das dritte durch den restlichen Körper. Und jetzt kommt das Verrückte: Wenn der Oktopus schwimmt, hört das dritte Herz einfach auf zu schlagen. Deshalb krabbeln Oktopusse lieber, als zu schwimmen. Schwimmen ist für sie schlicht anstrengend.

Ich habe letzte Woche etwas ausprobiert, das mein Leben verändert hat. Ich habe mein Handy jeden Abend um neun Uhr in die Küche gelegt. Nur das. Nach drei Tagen habe ich wieder Bücher gelesen. Nach einer Woche bin ich ohne Wecker aufgewacht. Ich sage dir ehrlich: Ich hätte nie gedacht, dass eine so kleine Regel so viel ausmacht.

Hier ist der größte Fehler, den fast jeder beim Sparen macht. Die meisten sparen das, was am Monatsende übrig bleibt. Aber am Monatsende bleibt nie etwas übrig. Dreh es um. Überweise am ersten Tag des Monats zehn Prozent auf ein zweites Konto, automatisch. Was du nicht siehst, gibst du nicht aus. So einfach ist das.

Kurze Frage: Was glaubst du, wie lange ein Blitz dauert? Eine Sekunde? Eine halbe? Falsch. Ein einzelner Blitz dauert oft nur etwa dreißig Mikrosekunden. Das, was wir als Flackern sehen, sind mehrere Entladungen hintereinander durch denselben Kanal. Unser Auge ist einfach zu langsam, um sie auseinanderzuhalten.

Und zum Schluss mein liebster Trick für Präsentationen. Fang niemals mit "Hallo, ich bin" an. Fang mit einer Frage an, die dein Publikum nicht beantworten kann. Sobald jemand über eine Frage nachdenkt, hört er dir zu. Probier es beim nächsten Mal aus und schreib mir, was passiert ist.
'@

Write-Host '==> generating speech with the Windows speech engine ...' -ForegroundColor Cyan
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'de-*' } | Select-Object -First 1
if ($voice) { $synth.SelectVoice($voice.VoiceInfo.Name) } else { Write-Host '   no German voice installed, using the default voice' -ForegroundColor Yellow }
$synth.Rate = 0
$synth.SetOutputToWaveFile($wav)
$synth.Speak($script)
$synth.Dispose()

Write-Host '==> rendering test video (1920x1080, animated waveform) ...' -ForegroundColor Cyan
ffmpeg -hide_banner -loglevel error -y -f lavfi -i 'color=c=0x14213d:s=1920x1080:r=30' -i $wav `
    -filter_complex '[1:a]showwaves=s=1920x600:mode=cline:colors=0x00d2ff|0xffb703[w];[0:v][w]overlay=0:240:shortest=1,format=yuv420p[v]' `
    -map '[v]' -map 1:a -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 160k -shortest $mp4
if ($LASTEXITCODE -ne 0) { throw 'ffmpeg failed to render the test video' }

Write-Host '==> running AutoClipper on it (transcription, selection, editing, QA, export) ...' -ForegroundColor Cyan
$start = Get-Date
$json = autoclipper process $mp4 --confirmed --title 'AutoClipper Selbsttest' | Out-String
$code = $LASTEXITCODE
Write-Host $json
if ($code -ne 0) { throw "self test failed (exit $code)" }
$result = $json | ConvertFrom-Json
$n = @($result.clips).Count
Write-Host ("==> self test passed: {0} clip(s) in {1:n1} min -> {2}" -f $n, ((Get-Date) - $start).TotalMinutes, $result.project) -ForegroundColor Green
foreach ($c in $result.clips) { Write-Host ("    #{0} score {1} qa {2}  {3}" -f $c.rank, $c.score, $c.qa, $c.title) }

if ($YouTubeUrl) {
    Write-Host '==> YouTube test: rights check ...' -ForegroundColor Cyan
    autoclipper rights $YouTubeUrl
    if ($LASTEXITCODE -eq 3) { Write-Host '    not Creative Commons / not your channel: skipped (no permission)' -ForegroundColor Yellow; return }
    autoclipper process $YouTubeUrl
}
