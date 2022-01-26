#!/bin/sh
export MULTI_LANGS="amharic-arabic-azerbaijani-bengali-burmese-chinese_simplified-chinese_traditional-english-french-gujarati-hausa-hindi-igbo-indonesian-japanese-kirundi-korean-kyrgyz-marathi-nepali-oromo-pashto-persian-pidgin-portuguese-punjabi-russian-scottish_gaelic-serbian_cyrillic-serbian_latin-sinhala-somali-spanish-swahili-tamil-telugu-thai-tigrinya-turkish-ukrainian-urdu-uzbek-vietnamese-welsh-yoruba"
export NPROC_PER_NODE=4

export lr=1e-3
export LM_lr=5e-5

python -m torch.distributed.launch \
	--nproc_per_node=$NPROC_PER_NODE \
    --master_port=30000 \
    "./pipeline_prefix_tuning.py" \
    --model_name_or_path "/projects/tir4/users/yiweiq/ckp/mT5_multilingual_XLSum" \
    --data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/multilingual" \
    --output_dir "/projects/tir4/users/yiweiq/PrefixTuning_data/xlsum/multi_44_all_from_xlsum_ckpt_private/mt5-base/lr_${lr}_LM_lr_${LM_lr}_no_warm"  \
    --predict_with_generate \
    --per_device_train_batch_size 8 \
    --use_encoder_prefix \
    --use_self_prefix \
    --use_cross_prefix \
    --overwrite_output_dir \
    --max_steps 75000 \
    --logging_steps 100 \
    --save_steps 5000 \
    --do_train \
    --learning_rate $lr \
    --preseqlen 200 \
    --upsampling_factor 0.5 \
    --logging_first_step \
    --not_freeze_lmodel \
    --adafactor \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --multi_languages $MULTI_LANGS \
    --private_prefix \
    --mid_dim 200 \
    --different_lr \
    --learning_rate_LM $LM_lr 

    
