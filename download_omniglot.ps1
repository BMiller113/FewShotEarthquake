# download_omniglot.ps1
$DataDir = "data/omniglot/data"

# 1) Create target directory
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

# 2) Download zips (equivalent to wget -O ...)
Invoke-WebRequest `
  -Uri "https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true" `
  -OutFile "images_background.zip"

Invoke-WebRequest `
  -Uri "https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true" `
  -OutFile "images_evaluation.zip"

# 3) Unzip into target directory (equivalent to unzip ... -d $DATADIR)
Expand-Archive -Force -Path "images_background.zip" -DestinationPath $DataDir
Expand-Archive -Force -Path "images_evaluation.zip" -DestinationPath $DataDir

# 4) Flatten directory structure:
#    bash: mv $DATADIR/images_background/* $DATADIR/
#          mv $DATADIR/images_evaluation/* $DATADIR/
$bgDir = Join-Path $DataDir "images_background"
$evDir = Join-Path $DataDir "images_evaluation"

if (Test-Path $bgDir) {
  Get-ChildItem -Path $bgDir -Force | ForEach-Object {
    Move-Item -Force -Path $_.FullName -Destination $DataDir
  }
}

if (Test-Path $evDir) {
  Get-ChildItem -Path $evDir -Force | ForEach-Object {
    Move-Item -Force -Path $_.FullName -Destination $DataDir
  }
}

# 5) Remove now-empty directories (equivalent to rmdir ...)
if (Test-Path $bgDir) { Remove-Item -Recurse -Force $bgDir }
if (Test-Path $evDir) { Remove-Item -Recurse -Force $evDir }

Write-Host "Omniglot downloaded and extracted to: $DataDir"
