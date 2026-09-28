$ErrorActionPreference = 'SilentlyContinue'
Set-Location 'D:\linkchain'

$portCheck = Get-NetTCPConnection -LocalPort 8015 -ErrorAction SilentlyContinue
if (-not $portCheck) {
    Start-Process -FilePath 'py.exe' -ArgumentList '-m','uvicorn','embaded.app:app','--host','127.0.0.1','--port','8015' -WorkingDirectory 'D:\linkchain' -WindowStyle Hidden
}
