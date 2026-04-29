#!/bin/bash

STUDENT_CHECKPOINT=

python eval.py \
    --model_name sedd \
    --scheduler_name euler \
    --device cuda:0 \
    --pretrained_model_path \
    --student_checkpoint ${STUDENT_CHECKPOINT} \
    --student_nfe 8 \
    --num_samples 10 \
    --batch_size 4 \
    --eval_model_name gpt2-large \
    --eval_batch_size 4 \
    --seed 42 \
    --output_file \
    --save_samples ./generated_samples.pt

