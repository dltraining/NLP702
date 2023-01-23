#!/bin/bash

#continue pre-training RoBERTa by take roberta checkpoint
python3 ./run_language_modeling.py \
        --gradient_accumulation_steps 8 \
        --train_data_file training_sample.txt \
        --output_dir ./ckpt_model/ \
        --model_type roberta \
        --logging_dir ./runs/ \
        --mlm \
        --fp16 \
        --num_workers 0 \
        --model_name_or_path ./roberta/ \
        --config_name ./roberta \
        --tokenizer_name ./roberta \
        --do_train \
        --block_size 256 \
        --learning_rate 5e-5 \
        --num_train_epochs 5 \
        --save_total_limit 5 \
        --per_gpu_train_batch_size 32 \
        --seed 42