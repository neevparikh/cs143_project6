#!/bin/bash

if [[ $# -ne 2 ]]; then
  >&2 echo "2 arguments are required"
  exit 1
fi

ffmpeg -i "$1" -vf scale=512:1024 "$PWD/$2%05d.png"
