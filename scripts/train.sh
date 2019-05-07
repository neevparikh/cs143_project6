#!/bin/bash
python3 src/train.py --dataroot="data/" --label_nc=18  --no_instance --verbose --loadSize=512 --no_flip --serial_batches $@
