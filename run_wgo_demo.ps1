param(
    [switch]$Prepare,
    [switch]$RecordApi,
    [string]$Model = 'gemini-3.8-flash',
    [string]$DataRoot,
    [string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$DemoPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$DemoScript = Join-Path $ProjectRoot 'examples\wgo_ego\run_demo.py'
if (-not (Test-Path -LiteralPath $DemoPython)) { throw 'The demo Python environment was not found.' }
# Python loads .env in the project root, then prompts only if no key is present.
$DemoArguments = @('--model', $Model)
if ($Prepare) { $DemoArguments += '--prepare' }
if ($RecordApi) { $DemoArguments += '--record-api' }
if ($DataRoot) { $DemoArguments += @('--data-root', $DataRoot) }
if ($OutputRoot) { $DemoArguments += @('--output-root', $OutputRoot) }
& $DemoPython $DemoScript @DemoArguments
exit $LASTEXITCODE
