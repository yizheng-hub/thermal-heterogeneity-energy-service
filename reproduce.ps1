$ErrorActionPreference = "Stop"
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvProgram = if ($uvCommand) { $uvCommand.Source } else { Join-Path $PSScriptRoot "intermediate_files/runtime/uv.exe" }
if (!(Test-Path -LiteralPath $uvProgram)) { throw "Install uv before running reproduction." }
& $uvProgram run --frozen --project $PSScriptRoot python (Join-Path $PSScriptRoot "reproduce.py") @args
if ($LASTEXITCODE -ne 0) { throw "Scientific reproduction failed." }
