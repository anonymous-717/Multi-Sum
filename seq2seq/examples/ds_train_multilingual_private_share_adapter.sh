#!/bin/bash
export MULTI_LANGS="amharic-arabic-azerbaijani-bengali-burmese-chinese_simplified-chinese_traditional-english-french-gujarati-hausa-hindi-igbo-indonesian-japanese-kirundi-korean-kyrgyz-marathi-nepali-oromo-pashto-persian-pidgin-portuguese-punjabi-russian-scottish_gaelic-serbian_cyrillic-serbian_latin-sinhala-somali-spanish-swahili-tamil-telugu-thai-tigrinya-turkish-ukrainian-urdu-uzbek-vietnamese-welsh-yoruba"

export NPROC_PER_NODE=4
export lr=1e-2
export LM_lr=2e-4

python -m torch.distributed.launch \
        --nproc_per_node=$NPROC_PER_NODE \
    "./pipeline_adaptor.py" \
    --model_name_or_path "/projects/tir4/users/yiweiq/ckp/mT5_multilingual_XLSum" \
    --data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/multilingual" \
    --output_dir "/projects/tir4/users/yiweiq/Adaptor_data/multi/multi_44_all_from_xlsum_private/lr_${lr}_LM_lr_${LM_lr}_no_warm" \
    --learning_rate $lr \
    --gradient_accumulation_steps 4 \
    --max_steps 75000 \
    --logging_steps 100 \
    --save_steps 5000 \
    --adafactor \
    --per_device_train_batch_size 8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train \
    --logging_first_step \
    --upsampling_factor 0.5 \
    --use_adaptor \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --not_freeze_lmodel \
    --adaptor_mid_dim 300 \
    --private_adapter \
    --multi_languages $MULTI_LANGS \
    --different_lr \
    --learning_rate_LM $LM_lr



