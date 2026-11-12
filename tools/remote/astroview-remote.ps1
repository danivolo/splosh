<#
.SYNOPSIS
  Start `splosh serve` on a remote login node over ssh, forward its port, open the page; closing this window (or Ctrl+C) stops the server.

.EXAMPLE
  .\splosh-remote.ps1                                                 # no run: the page opens a folder picker (your home directory on the remote)
  .\splosh-remote.ps1 -Run '~/runs/myrun/output'                      # 4 cores, at most 8 from the page
  .\splosh-remote.ps1 -Run '~/runs/a ~/runs/b' -Cores 2 -MaxCores 4 -Extra '--layouts ~/layouts --analysis ~/runs/a/analysis'

.NOTES
  - The token and the port are made here, so the URL is known before the server starts; ssh -L forwards the same port on both ends.
  - ssh -tt gives the remote command a terminal: when ssh ends, the server gets a hangup and exits. --exit-when-orphaned and --idle-exit
    stop it as well if the hangup is lost (a dropped connection).
  - Cores: numba's pool is NUMBA_NUM_THREADS=MaxCores, fixed at start; the server begins with -Cores and the page's "cores" box changes the
    count between 1 and MaxCores without a restart. The process runs under nice 10 (a login node is shared).
  - -Slurm 'srun options' lets the page's long jobs (movie, prefetch all) take a free compute node at once, e.g. -Slurm '-c 20 --mem=32G -t 01:00:00'
    (always give --mem); with no free node they run on the login node at nice 10 with the viewer's cores. -Ffmpeg PATH when ffmpeg is not on the remote PATH.
  - -Root limits what the folder picker may browse (default: the remote home directory).
  - -Splosh is the remote command that starts splosh, e.g. '$(astro pyenv path splosh)/bin/splosh' or a full path.
#>
param(
  [string]$Run = "",
  [string]$SshHost = "cluster",
  [int]$Cores = 4,
  [int]$MaxCores = 8,
  [int]$IdleMinutes = 120,
  [string]$Splosh = "splosh",
  [string]$Extra = "",
  [string]$Root = "",
  [string]$Slurm = "",
  [string]$Ffmpeg = ""
)
$ErrorActionPreference = "Stop"
if ($Cores -gt $MaxCores) { $MaxCores = $Cores }
$port = Get-Random -Minimum 20000 -Maximum 60000
$token = -join ((48..57) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })
$remote = "NUMBA_NUM_THREADS=$MaxCores exec nice -n 10 $Splosh serve $Run$(if ($Root) { " --root $Root" })$(if ($Slurm) { " --slurm '$Slurm'" })$(if ($Ffmpeg) { " --ffmpeg $Ffmpeg" }) --port $port --token $token --threads $Cores --exit-when-orphaned --idle-exit $IdleMinutes $Extra"
$Host.UI.RawUI.WindowTitle = "splosh on $SshHost :$port  (close to stop)"
Write-Host "ssh $SshHost : $remote" -ForegroundColor DarkGray
Write-Host "close this window or press Ctrl+C to stop the server" -ForegroundColor Yellow
$opened = $false
& ssh -tt -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "${port}:localhost:${port}" $SshHost $remote 2>&1 | ForEach-Object {
  Write-Host ("`r" + "$_".TrimEnd("`r"))        # ssh -tt puts the console in raw mode: a line feed alone does not return to column 0
  if (-not $opened -and "$_" -match 'open\s+(http://\S+)') {
    Start-Process $Matches[1]
    Set-Clipboard -Value $Matches[1]
    $opened = $true
    Write-Host "`ropened in the browser (URL also on the clipboard)" -ForegroundColor Green
  }
}
Write-Host "`rssh ended (exit $LASTEXITCODE); the server stops with it." -ForegroundColor Yellow
if (-not $opened) { Read-Host "the server did not start; press Enter to close" }
