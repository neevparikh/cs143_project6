#!/bin/bash
python3 src/test.py --dataroot "data/" --label_nc=18  --no_instance --verbose --nThreads=0 --loadSize=512 --serial_batches $@
