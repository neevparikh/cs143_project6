#!/bin/bash
python3 src/train.py --dataroot="data/" --label_nc=18  --no_instance --verbose --nThreads=0 --no_html --loadSize 256 $@
