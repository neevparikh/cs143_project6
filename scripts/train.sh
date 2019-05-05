#!/bin/bash
python3 src/train.py --dataroot="data.bak/" --label_nc=18  --no_instance --verbose --nThreads=0 --loadSize 256 $@
