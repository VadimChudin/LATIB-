$symbols = Get-ChildItem "d:\LAITB 2.0\data\epicenters" -Filter "*_epicenters.json" | ForEach-Object {
    $_.Name -replace '_epicenters\.json$', ''
}

$results = @()
$total = $symbols.Count
$i = 0

foreach ($symbol in $symbols) {
    $i++
    Write-Host "`n[$i/$total] $symbol" -ForegroundColor Cyan
    
    $output = & "d:\LAITB 2.0\rust_engine\target\release\aegis_engine.exe" optimize-ticks --symbol $symbol --direction ALL --generations 200 2>&1
    
    # Extract key lines
    $trainLine = $output | Select-String "TRAIN SET"
    $testLine = $output | Select-String "TEST SET"
    $savedLine = $output | Select-String "Saved .* trades"
    
    if ($trainLine) { Write-Host "  $trainLine" -ForegroundColor Green }
    if ($testLine) { Write-Host "  $testLine" -ForegroundColor Yellow }
    if ($savedLine) { Write-Host "  $savedLine" }
    
    $results += "$symbol|$trainLine|$testLine"
}

Write-Host "`n`n========== SUMMARY ==========" -ForegroundColor Magenta
foreach ($r in $results) {
    Write-Host $r
}
