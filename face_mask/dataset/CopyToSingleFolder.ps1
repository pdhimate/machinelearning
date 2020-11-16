$ErrorActionPreference = 'Stop'
$Src = "C:\Work\GitHub\machinelearning2\face_mask\dataset\raw"
$Dst = "C:\Work\GitHub\machinelearning2\face_mask\dataset\raw\NewImages"

$count = 0

foreach($file in Get-ChildItem $Src -Recurse -File)
{
    $guid = New-Guid
    $newName = Join-Path $Dst ($guid.ToString() + $file.Extension)
   
    Copy-Item -Path $file.PSPath -Destination $newName 
    $count++
}
Write-Output "Copied $count files"

