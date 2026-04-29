#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
python train_teacher_generate.py \
    --model_name sedd \
    --scheduler_name euler \
    --device cuda:0 \
    --pretrained_model_path \
    --lsd_num_samples 1 \
    --lsd_teacher_nfe 256 \
    --student_nfe 8 \
    --epochs 16 \
    --batch_size 1 \
    --offline_data_dir \
    --seed 42

