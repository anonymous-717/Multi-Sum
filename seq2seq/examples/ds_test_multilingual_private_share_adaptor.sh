#!/bin/sh
export lr=1e-3

export MULTI_LANGS="amharic-arabic-azerbaijani-bengali-burmese-chinese_simplified-chinese_traditional-english-french-gujarati-hausa-hindi-igbo-indonesian-japanese-kirundi-korean-kyrgyz-marathi-nepali-oromo-pashto-persian-pidgin-portuguese-punjabi-russian-scottish_gaelic-serbian_cyrillic-serbian_latin-sinhala-somali-spanish-swahili-tamil-telugu-thai-tigrinya-turkish-ukrainian-urdu-uzbek-vietnamese-welsh-yoruba"


export model="lr_1e-2_LM_lr_2e-4_no_warm"

python pipeline_adaptor.py \
    --model_name_or_path "/projects/tir4/users/yiweiq/Adaptor_data/multi/multi_44_all_from_xlsum_private/${model}/checkpoint-60000" \
    --data_dir "/projects/tir4/users/yiweiq/data/XLSum_input/individual/amharic/" \
    --output_dir "/projects/tir4/users/yiweiq/${model}/amharic/adapter_private_with_self_adapter_diff_lr/checkpoint-60000" \
    --rouge_lang "amharic" \
    --predict_with_generate \
    --do_predict \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --use_adaptor \
    --adaptor_mid_dim 300 \
    --private_adapter \
    --multi_languages $MULTI_LANGS 
