# Quick Image Renamer for CTVO Dataset
# This script renames your images systematically for easy organization

Write-Host "========================================"
Write-Host "CTVO Image Renamer"
Write-Host "========================================"

# Check if temp_organization folder exists
if (-not (Test-Path "temp_organization")) {
    Write-Host "ERROR: temp_organization folder not found!"
    Write-Host "Please create it and copy your images there first"
    Read-Host "Press Enter to exit"
    exit 1
}

# Get all JPG images
$images = Get-ChildItem "temp_organization\*.jpg" | Sort-Object Name
if ($images.Count -eq 0) {
    Write-Host "ERROR: No JPG images found in temp_organization folder!"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Found $($images.Count) images to rename"
Write-Host ""

# Rename images systematically
for ($i = 0; $i -lt $images.Count; $i++) {
    $person = [math]::Floor($i / 8) + 1
    $view = ($i % 8) + 1
    $newName = "person{0:D3}_view{1:D2}.jpg" -f $person, $view
    
    $oldPath = $images[$i].FullName
    $newPath = Join-Path "temp_organization" $newName
    
    Rename-Item $oldPath $newPath
    Write-Host "Renamed: $($images[$i].Name) → $newName"
}

Write-Host ""
Write-Host "========================================"
Write-Host "✅ RENAMING COMPLETE!"
Write-Host "========================================"
Write-Host ""
Write-Host "Your images are now named:"
Write-Host "person001_view01.jpg, person001_view02.jpg, ... person001_view08.jpg"
Write-Host "person002_view01.jpg, person002_view02.jpg, ... person002_view08.jpg"
Write-Host "... and so on for all 12 people"
Write-Host ""
Write-Host "Next step: Run auto_organize_images.bat to organize them into directories"
Write-Host ""

Read-Host "Press Enter to exit"
