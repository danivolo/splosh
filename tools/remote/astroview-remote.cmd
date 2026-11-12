@echo off
rem double-click wrapper: splosh-remote.cmd -Run "~/runs/myrun/output" [-Cores 4] [-MaxCores 8] [-SshHost cluster]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0splosh-remote.ps1" %*
