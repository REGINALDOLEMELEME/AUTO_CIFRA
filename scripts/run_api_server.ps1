param(
  [string]$HostAddress = "127.0.0.1",
  [int]$Port = 8000
)

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here

& "$Root\.venv\Scripts\python.exe" -m uvicorn "app.api:create_app" --factory --host $HostAddress --port $Port --app-dir $Root
