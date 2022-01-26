#!/bin/bash
export num_train_epochs=15
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export PER_DEVICE_EVAL_BATCH_SIZE=24
export GRADIENT_ACC=1

export lr=1e-3

export language="amharic"
export num_train=5761


export output_dir="/projects/tir4/users/yiweiq/Adaptor_data/single_languae_from_mt5_base/lr_${lr}_ada_no_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}/${language}/"

python pipeline_adaptor.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/individual/${language}" \
    --output_dir  $output_dir \
    --learning_rate $lr \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --num_train_epochs $num_train_epochs \
    --logging_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/num_train_epochs)) \
    --save_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/num_train_epochs)) \
    --eval_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/num_train_epochs)) \
    --adafactor \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE  \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --predict_with_generate \
    --do_train \
    --do_eval \
    --rouge_lang $language \
    --logging_first_step \
    --metric_for_best_model rouge2 \
    --greater_is_better True \
    --use_adaptor \
    --n_val 500 

python choose_best_on_val_set.py $output_dir

python pipeline_adaptor.py \
    --model_name_or_path "/projects/tir4/users/yiweiq/Adaptor_data/single_languae_from_mt5_base/lr_${lr}_ada_no_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}/${language}/best_ckpt/" \
    --data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/individual/${language}" \
    --output_dir  "/projects/tir4/users/yiweiq/Adaptor_data/single_languae_from_mt5_base/lr_${lr}_ada_no_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}/${language}/test/" \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_predict \
    --rouge_lang $language \
    --use_adaptor 

        

