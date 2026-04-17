param(
  [string]$VenvPath = ".venv"
)

python -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "Python environment ready."
Write-Host "Install external tools manually if missing: ffmpeg, sonic-annotator, chordino plugin."
