#!/bin/bash

blocks=(
  "1024 1 1"
  "512 2 1"
  "256 2 2"
  "256 4 1"
  "128 4 2"
  "128 8 1"
  "64 16 1"
  "64 4 4"
  "32 8 4"
  "32 32 1"
  "16 8 8"
  "16 16 4"
)

for block in "${blocks[@]}"; do
  read -r BX BY BZ <<< "$block"
  for run in {1..3}; do
    mojo laplacian.mojo --block $BX $BY $BZ >> res.txt
  done
done

echo "Done"
