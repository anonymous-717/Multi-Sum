#!/bin/sh
export model="lr_1e-3_LM_lr_5e-5_no_warm"
export MULTI_LANGS="amharic-arabic-azerbaijani-bengali-burmese-chinese_simplified-chinese_traditional-english-french-gujarati-hausa-hindi-igbo-indonesian-japanese-kirundi-korean-kyrgyz-marathi-nepali-oromo-pashto-persian-pidgin-portuguese-punjabi-russian-scottish_gaelic-serbian_cyrillic-serbian_latin-sinhala-somali-spanish-swahili-tamil-telugu-thai-tigrinya-turkish-ukrainian-urdu-uzbek-vietnamese-welsh-yoruba"

python pipeline_prefix_tuning.py \
    --model_name_or_path google/mt5-base \
    --prefixModel_name_or_path "/projects/tir4/users/yiweiq/PrefixTuning_data/xlsum/multi_44_all_from_xlsum_ckpt_private/mt5-base/${model}/checkpoint-65000" \
    --data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/individual/amharic" \
    --output_dir "/projects/tir4/users/yiweiq/PrefixTuning_data/xlsum/multi_44_all_from_xlsum_ckpt_private/mt5-base/${model}/test/amharic" \
    --rouge_lang "amharic" \
    --predict_with_generate \
    --load_whole_model \
    --do_predict \
    --per_device_eval_batch_size 24 \
    --overwrite_output_dir \
    --use_cross_prefix \
    --use_encoder_prefix \
    --use_self_prefix \
    --preseqlen 200 \
    --mid_dim 200 \
    --private_prefix \
    --multi_languages $MULTI_LANGS
