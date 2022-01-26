#!/bin/bash
export num_train_epochs=15
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export PER_DEVICE_EVAL_BATCH_SIZE=24
export GRADIENT_ACC=1

export lr=2e-4

export language="amharic"
export num_train=5761
    
export output_dir="/projects/tir4/users/yiweiq/PrefixTuning_data/single_languae_from_mt5_base/prefix_tune/lr_${lr}_ada_all_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}/${language}/"

python ./pipeline_prefix_tuning.py \
--model_name_or_path  "google/mt5-base" \
--data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/individual/${language}/" \
--output_dir $output_dir \
--rouge_lang $language \
--predict_with_generate \
--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
--per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE  \
--use_encoder_prefix \
--use_self_prefix \
--use_cross_prefix \
--overwrite_output_dir \
--num_train_epochs $num_train_epochs \
--logging_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/num_train_epochs)) \
--save_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/num_train_epochs)) \
--eval_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/num_train_epochs)) \
--evaluation_strategy steps \
--do_train \
--do_eval \
--metric_for_best_model rouge2 \
--greater_is_better True \
--learning_rate $lr \
--max_target_length 84 \
--val_max_target_length 84 \
--n_val 500 \
--gradient_accumulation_steps $GRADIENT_ACC\
--logging_first_step \
--adafactor \
--warmup_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/10)) \
--weight_decay 0.01 \
--label_smoothing_factor 0.1

python choose_best_on_val_set.py $output_dir

python pipeline_prefix_tuning.py \
--model_name_or_path  "/projects/tir4/users/yiweiq/ckp/mT5_multilingual_XLSum" \
--prefixModel_name_or_path "/projects/tir4/users/yiweiq/PrefixTuning_data/single_languae_from_mt5_base/prefix_tune/lr_${lr}_ada_all_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}/${language}/best_ckpt/" \
--data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/individual/${language}/" \
--output_dir "/projects/tir4/users/yiweiq/PrefixTuning_data/single_languae_from_mt5_base/prefix_tune/lr_${lr}_ada_all_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}/${language}/test/" \
--load_whole_model \
--rouge_lang $language \
--predict_with_generate \
--per_device_eval_batch_size 24 \
--do_predict \
--use_encoder_prefix \
--use_self_prefix \
--use_cross_prefix \
--overwrite_output_dir 
