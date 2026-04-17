param(
  [int]$Port = 8000,
  [string]$BindAddress = "127.0.0.1"
)

$venvPython = ".\.venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
  & $venvPython -m app.upload_server --host $BindAddress --port $Port
} else {
  python -m app.upload_server --host $BindAddress --port $Port
}
