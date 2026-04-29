#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python train_student_distill.py \
    --model_name sedd \
    --scheduler_name euler \
    --device cuda:0 \
    --pretrained_model_path  \
    --offline_data_dir  \
    --output_dir \
    --lr 1e-2 \
    --batch_size 1 \
    --student_nfe 8 \
    --eps 1e-3 \
    --ema_decay 0.9999 \
    --load_epoch_start 0 \
    --load_epoch_end -1

