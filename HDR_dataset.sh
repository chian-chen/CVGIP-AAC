#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="./HDR_raw/HDR_png"
DEST_DIR="./HDR_datas"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# List of files to copy
files=(
  "507.png"
  "BalancedRock.png"
  "BloomingGorse(1).png"
  "BloomingGorse(2).png"
  "CemeteryTree(1).png"
  "CemeteryTree(2).png"
  "DelicateFlowers.png"
  "DevilsTower.png"
  "Exploratorium(1).png"
  "Flamingo.png"
  "FourCornersStorm.png"
  "Frontier.png"
  "GoldenGate(1).png"
  "HalfDomeSunset.png"
  "HallofFame.png"
  "HancockKitchenInside.png"
  "HooverDam.png"
  "JesseBrownsCabin.png"
  "LetchworthTeaTable(1).png"
  "LetchworthTeaTable(2).png"
  "MasonLake(2).png"
  "MtRushmore(1).png"
  "MtRushmore(2).png"
  "MtRushmoreFlags.png"
  "OtterPoint.png"
  "Route66Museum.png"
  "SunsetPoint(2).png"
  "TheGrotto.png"
  "TupperLake(1).png"
  "WestBranchAusable(1).png"
  "WillyDesk.png"
)

# Copy each file
for file in "${files[@]}"; do
  cp "$SRC_DIR/$file" "$DEST_DIR/"
done

echo "All files have been copied to $DEST_DIR."
