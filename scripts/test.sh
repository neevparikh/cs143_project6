#!/bin/bash
python3 src/test.py --dataroot "data/" --no_instance --verbose --nThreads=0 --loadSize=512 $@
