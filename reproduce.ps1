$ErrorActionPreference = "Stop"
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (!$uvCommand) { throw "Install uv before running reproduction." }
& $uvCommand.Source run --frozen --project $PSScriptRoot python (Join-Path $PSScriptRoot "reproduce.py") @args
if ($LASTEXITCODE -ne 0) { throw "Scientific reproduction failed." }
