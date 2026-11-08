param([string]$Address)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
  if (!$Address) { $Address = Read-Host 'Computer Wi-Fi IPv4 address (see ipconfig)' }
  if ($Address -notmatch '^\d{1,3}(\.\d{1,3}){3}$') { throw 'IPv4 address required' }
  New-Item -ItemType Directory -Force -Path '.local' | Out-Null
  "[req]`ndistinguished_name=dn`n[dn]" | Set-Content -LiteralPath '.local/openssl.cnf' -Encoding ascii
  if (!(Test-Path -LiteralPath '.local/ca.pem')) {
    & openssl req -config .local/openssl.cnf -x509 -newkey rsa:2048 -nodes -keyout .local/ca-key.pem -out .local/ca.pem -sha256 -days 365 -subj '/CN=Three Phone IMU Local CA' -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' -addext 'keyUsage=critical,keyCertSign,cRLSign'
    if ($LASTEXITCODE -ne 0) { throw 'CA generation failed' }
  }
  @"
subjectAltName=IP:$Address,IP:127.0.0.1,DNS:localhost
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
"@ | Set-Content -LiteralPath '.local/server.ext' -Encoding ascii
  & openssl req -config .local/openssl.cnf -new -newkey rsa:2048 -nodes -keyout .local/server-key.pem -out .local/server.csr -subj '/CN=Three Phone IMU'
  if ($LASTEXITCODE -ne 0) { throw 'Server key generation failed' }
  & openssl x509 -req -in .local/server.csr -CA .local/ca.pem -CAkey .local/ca-key.pem -CAcreateserial -out .local/server.pem -days 90 -sha256 -extfile .local/server.ext
  if ($LASTEXITCODE -ne 0) { throw 'Server certificate generation failed' }
  & node scripts/setup.mjs $Address --existing-certs
  if ($LASTEXITCODE -ne 0) { throw 'Setup failed' }
} finally { Pop-Location }
