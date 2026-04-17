param(
  [string]$VenvPython = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "SilentlyContinue"

function Write-Check {
  param(
    [string]$Name,
    [bool]$Ok,
    [string]$Detail
  )
  $status = if ($Ok) { "OK" } else { "MISSING" }
  Write-Host ("[{0}] {1} - {2}" -f $status, $Name, $Detail)
}

Write-Host "AUTO_CIFRA Environment Diagnosis"
Write-Host "================================"

$pythonCmd = Get-Command python
$pythonDetail = if ($pythonCmd) { $pythonCmd.Source } else { "python not in PATH" }
Write-Check "python" ($null -ne $pythonCmd) $pythonDetail

$venvExists = Test-Path -LiteralPath $VenvPython
$venvDetail = if ($venvExists) { $VenvPython } else { "venv not found" }
Write-Check "venv python" $venvExists $venvDetail

$ffmpegCmd = Get-Command ffmpeg
$ffmpegDetail = if ($ffmpegCmd) { $ffmpegCmd.Source } else { "ffmpeg not in PATH" }
Write-Check "ffmpeg" ($null -ne $ffmpegCmd) $ffmpegDetail

$sonicCmd = Get-Command sonic-annotator
$localSonic = "tools\\sonic-annotator\\sonic-annotator-win64\\sonic-annotator.exe"
$hasLocalSonic = Test-Path -LiteralPath $localSonic
$sonicOk = ($null -ne $sonicCmd) -or $hasLocalSonic
$sonicDetail = if ($sonicCmd) { $sonicCmd.Source } elseif ($hasLocalSonic) { "$localSonic (local)" } else { "sonic-annotator not in PATH and local tool missing" }
Write-Check "sonic-annotator" $sonicOk $sonicDetail

$vampDir = "tools\\vamp-plugins"
$hasVampDir = Test-Path -LiteralPath $vampDir
$chordinoDll = Get-ChildItem -LiteralPath $vampDir -Filter "*nnls-chroma*.dll" -Recurse
$chordinoOk = $hasVampDir -and ($chordinoDll.Count -gt 0)
$chordinoDetail = if ($chordinoOk) { $chordinoDll[0].FullName } elseif ($hasVampDir) { "vamp folder exists but nnls-chroma dll not found" } else { "tools\\vamp-plugins folder missing" }
Write-Check "chordino plugin dll" $chordinoOk $chordinoDetail

if ($venvExists) {
  $fw = & $VenvPython -c "import faster_whisper; print('ok')" 2>$null
  $fwOk = ($LASTEXITCODE -eq 0)
  $fwDetail = if ($fwOk) { "import ok" } else { "not importable" }
  Write-Check "python package: faster-whisper" $fwOk $fwDetail

  $pd = & $VenvPython -c "import docx; print('ok')" 2>$null
  $pdOk = ($LASTEXITCODE -eq 0)
  $pdDetail = if ($pdOk) { "import ok" } else { "not importable" }
  Write-Check "python package: python-docx" $pdOk $pdDetail

  $lb = & $VenvPython -c "import librosa; print('ok')" 2>$null
  $lbOk = ($LASTEXITCODE -eq 0)
  $lbDetail = if ($lbOk) { "import ok" } else { "not importable" }
  Write-Check "python package: librosa" $lbOk $lbDetail
} else {
  Write-Check "python package: faster-whisper" $false "venv missing"
  Write-Check "python package: python-docx" $false "venv missing"
  Write-Check "python package: librosa" $false "venv missing"
}

Write-Host ""
Write-Host "Chordino plugin check:"
Write-Host "- Put Chordino/NNLS dll files in: tools\\vamp-plugins"
Write-Host "- Then test:"
Write-Host "  .\\tools\\sonic-annotator\\sonic-annotator-win64\\sonic-annotator.exe -l | findstr chordino"

Write-Host ""
Write-Host "Done."
