param([Parameter(Mandatory=$true)][string]$Address)
$ErrorActionPreference = 'Stop'
if ($Address -notmatch '^\d{1,3}(\.\d{1,3}){3}$') { throw 'Supply the computer IPv4 address' }
$imuPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (!$imuPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Open PowerShell as Administrator and rerun this script. It only allows this Node executable on TCP 8080/8443, this computer IP, and LocalSubnet peers.'
}
$imuNode = (Get-Command node.exe).Source
New-NetFirewallRule -DisplayName "Three Phone IMU ($Address)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080,8443 -LocalAddress $Address -RemoteAddress LocalSubnet -Program $imuNode -Profile Any | Format-Table DisplayName,Enabled,Action
