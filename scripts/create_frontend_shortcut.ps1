$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$interpreterPath = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$launcherPath = Join-Path $projectRoot 'scripts\launch_frontend.py'

if (-not (Test-Path -LiteralPath $interpreterPath -PathType Leaf)) {
    throw 'Project .venv\Scripts\pythonw.exe is missing.'
}
if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
    throw 'Frontend launcher source is missing.'
}

# Unicode names are constructed explicitly so Windows PowerShell 5.1 can read
# this script without depending on its default source-file encoding.
$shortcutName = -join ([char[]](0x62A4, 0x76EE, 0x955C, 0x8BC6, 0x522B))
$shortcutPath = Join-Path $projectRoot ($shortcutName + '.lnk')
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $interpreterPath
$shortcut.Arguments = '-X utf8 -m scripts.launch_frontend'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = 'Open the local goggles recognition workspace'
$shortcut.Save()

Write-Output $shortcutPath
