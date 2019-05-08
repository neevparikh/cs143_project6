#!/bin/bash

#change 2 to num gpus

python3 -m  torch.distributed.launch --nproc_per_node=4 src/train.py \
  --dataroot="data/" --label_nc=18  --no_instance --verbose --loadSize=512 \
  --no_flip --serial_batches --fp16 $@
