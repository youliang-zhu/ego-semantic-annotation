param(
    [switch]$Prepare,
    [switch]$RecordApi,
    [string]$Model = 'gemini-3.8-flash'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$DemoPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$DemoScript = Join-Path $ProjectRoot 'examples\wgo_ego\run_demo.py'
if (-not (Test-Path -LiteralPath $DemoPython)) { throw 'The demo Python environment was not found.' }
if ($Prepare) {
    & $DemoPython $DemoScript --prepare
    exit $LASTEXITCODE
}
# Python loads .env in the project root, then prompts only if no key is present.
$DemoArguments = @('--model', $Model)
if ($RecordApi) { $DemoArguments += '--record-api' }
& $DemoPython $DemoScript @DemoArguments
exit $LASTEXITCODE
