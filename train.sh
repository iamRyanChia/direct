#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONPATH=./:$PYTHONPATH

traindata=../__DATASET/fastmri/train
valdata=../__DATASET/fastmri/val

ls $traindata >projects/fastmri_multicoil/lists/train.lst
ls $valdata >projects/fastmri_multicoil/lists/val.lst

python tools/train_rim.py $traindata $valdata output \
    --name base --cfg projects/fastmri_multicoil/configs/base.yaml --num-gpus $((${#CUDA_VISIBLE_DEVICES} / 2 + 1)) \
    --num-workers 1 --mixed-precision
