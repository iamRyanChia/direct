#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=./:$PYTHONPATH

python tools/train_rim.py /DATA_Inter/fastMRI/part/multicoil_train /DATA_Inter/fastMRI/part/multicoil_val output \
    --name base --cfg projects/fastmri_multicoil/configs/base.yaml --num-gpus $((${#CUDA_VISIBLE_DEVICES} / 2 + 1)) \
    --num-workers 1 --mixed-precision
