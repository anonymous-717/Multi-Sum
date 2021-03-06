We use a modified fork of [xl-sum](https://github.com/csebuetnlp/xl-sum#license) for our experiments.

## Setup

```bash
$ git clone https://github.com/anonymous-717/Multi-Sum.git
$ cd Multi-Sum/seq2seq
$ conda create python==3.7.9 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -p ./env
$ conda activate ./env # or source activate ./env (for older versions of anaconda)
$ bash setup.sh 
```
* Use the newly created environment for running rest of the commands.

## Extracting data

Before running the extractor, place all the `.jsonl` files (`train`, `val`, `test`) for all the languages you want to work with, under a single directory (without any subdirectories). 

For example, to replicate our multilingual setup with all languages, run the following commands:

```bash
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fKxf9jAj0KptzlxUsI3jDbp4XLv_piiD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fKxf9jAj0KptzlxUsI3jDbp4XLv_piiD" -O XLSum_complete_v2.0.tar.bz2 && rm -rf /tmp/cookies.txt
$ tar -xjvf XLSum_complete_v2.0.tar.bz2
$ python extract_data.py -i XLSum_complete_v2.0/ -o XLSum_input/
```
This will create the source and target files for multilingual training within `XLSum_input/multilingual` and per language training and evaluation filepairs under `XLSum_input/individual/<language>`.


## Training & Evaluation
### Single language training
* For single language adapter-tuning on a single GPU, an example is as follows:
```bash
$ python pipeline_adaptor.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir #DATA_DIR \
    --output_dir #OUTPUT_DIR \
    --learning_rate 1e-3 \
    --num_train_epochs 15 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --adafactor \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 24  \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --predict_with_generate \
    --do_train \
    --do_eval \
    --rouge_lang "amharic" \
    --logging_first_step \
    --metric_for_best_model rouge2 \
    --greater_is_better True \
    --use_adaptor 
```

* For single language prefix-tuning on a single GPU, an example is as follows:
```bash
$ python ./pipeline_prefix_tuning.py \
    --model_name_or_path  "google/mt5-base" \
    --data_dir #DATA_DIR \
    --output_dir #OUTPUT_DIR \
    --rouge_lang "amharic" \
    --predict_with_generate \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 24  \
    --use_encoder_prefix \
    --use_self_prefix \
    --overwrite_output_dir \
    --num_train_epochs 15 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --do_train \
    --do_eval \
    --metric_for_best_model rouge2 \
    --greater_is_better True \
    --learning_rate 2e-4 \
    --logging_first_step \
    --adafactor 
```

* For single language full fine-tuning on a single GPU, an example is as follows:
```bash
$ python pipeline.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir #DATA_DIR \
    --output_dir  #OUTPUT_DIR \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 15 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --adafactor \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 24  \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --predict_with_generate \
    --do_train \
    --do_eval \
    --rouge_lang "amharic" \
    --logging_first_step \
    --metric_for_best_model rouge2 \
    --greater_is_better True 
```
For a detailed example, refer to [adapter_tuning_and_test.sh](examples/adapter_tuning_and_test.sh), [prefix_tuning_and_test.sh](examples/prefix_tuning_and_test.sh) and [full_fine_tuning_and_test](examples/full_fine_tuning_and_test.sh).

### Single language test
* To calculate rouge scores on test sets using a single language adapter-tuning model, use the following snippet:
```bash
$ python pipeline_adaptor.py \
    --model_name_or_path #MODEL_NAME \
    --data_dir #DATA_DIR \
    --output_dir #OUTPUT_DIR \
    --per_device_eval_batch_size 24 \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_predict \
    --rouge_lang "amharic" \
    --use_adaptor 
```

* To calculate rouge scores on test sets using a single language prefix-tuning model, use the following snippet:
```bash
$ python pipeline_prefix_tuning.py \
    --model_name_or_path  #MODEL_NAME \
    --prefixModel_name_or_path #PREFIX_MODEL_NAME \
    --data_dir #DATA_DIR \
    --output_dir #OUTPUT_DIR \
    --load_whole_model \
    --rouge_lang "amharic" \
    --predict_with_generate \
    --per_device_eval_batch_size 24 \
    --do_predict \
    --use_encoder_prefix \
    --use_self_prefix \
    --overwrite_output_dir 
```

* To calculate rouge scores on test sets using a single language full fine-tuning model, use the following snippet:
```bash
$ python pipeline.py \
    --model_name_or_path #MODEL_NAME \
    --data_dir #DATA_DIR \
    --output_dir  #OUTPUT_DIR \
    --per_device_eval_batch_size 24 \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_predict \
    --rouge_lang "amharic"
```
For a detailed example, refer to [ds_test_multilingual_private_share_adaptor.sh](examples/ds_test_multilingual_private_share_adaptor.sh) and [ds_test_multilingual_private_share_prefix.sh](examples/ds_test_multilingual_private_share_prefix.sh).

### Multilingual training
* For multilingual training on a single GPU, an example is as follows:
```bash
$ python ./pipeline.py \
    --model_name_or_path #Model_Name_or_Path \
    --data_dir #Data_Dir \
    --output_dir #Output_Dir \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 4 \
    --max_steps 50000 \
    --logging_steps 100 \
    --save_steps 5000 \
    --adafactor \
    --per_device_train_batch_size 8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train \
    --logging_first_step \
    --upsampling_factor 0.5 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
```

* For multilingual private-share adapter training on a single GPU, an example is as follows:
```bash
$ python ./pipeline_adaptor.py \
    --model_name_or_path #Model_Name_or_Path \
    --data_dir #Data_Dir \
    --output_dir #Output_Dir \
    --learning_rate 1e-2 \
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
    --multi_languages #Languages \
    --different_lr \
    --learning_rate_LM 2e-4
```
* For multilingual private-share prefix training on a single GPU, an example is as follows:
```bash
$ python ./pipeline_prefix_tuning.py \
    --model_name_or_path #Model_Name_or_Path \
    --data_dir #Data_Dir \
    --output_dir #Output_Dir  \
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
    --learning_rate 1e-3 \
    --preseqlen 200 \
    --upsampling_factor 0.5 \
    --logging_first_step \
    --not_freeze_lmodel \
    --adafactor \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --multi_languages #Languages \
    --private_prefix \
    --mid_dim 200 \
    --different_lr \
    --learning_rate_LM 5e-5 
```
To replicate our setup on 4 GPUs using SLURM, refer to [distributed_trainer.sh](distributed_trainer.sh), [ds_train_multilingual_private_share_adapter.sh](examples/ds_train_multilingual_private_share_adapter.sh) and [ds_train_multilingual_private_share_prefix.sh](examples/ds_train_multilingual_private_share_prefix.sh) 


### Multilingual Test
* To calculate rouge scores on test sets (for example on `amharic`) using a trained multilingual model, use the following snippet:
```bash
$ python pipeline.py \
    --model_name_or_path #Model_Name_or_Path \
    --data_dir #Data_Dir \
    --output_dir #Output_Dir \
    --rouge_lang "amharic" \
    --predict_with_generate \
    --do_predict \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir 
```

* To calculate rouge scores on test sets (for example on `amharic`) using a trained multilingual private-share adapter model, use the following snippet:

```bash
$ python pipeline_adaptor.py \
    --model_name_or_path #Model_Name_or_Path \
    --data_dir #Data_Dir \
    --output_dir #Output_Dir \
    --rouge_lang "amharic" \
    --predict_with_generate \
    --do_predict \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --use_adaptor \
    --adaptor_mid_dim 300 \
    --private_adapter \
    --multi_languages #Languages 
```

* To calculate rouge scores on test sets (for example on `amharic`) using a trained multilingual private-share prefix model, use the following snippet:

```bash
$ python pipeline_prefix_tuning.py \
    --model_name_or_path google/mt5-base \
    --prefixModel_name_or_path #Model_Name_or_Path \
    --data_dir #Data_Dir \
    --output_dir #Output_Dir \
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
    --multi_languages #Languages
```

For a detailed example, refer to [ds_test_multilingual_private_share_adaptor.sh](examples/ds_test_multilingual_private_share_adaptor.sh) and [ds_test_multilingual_private_share_prefix.sh](examples/ds_test_multilingual_private_share_prefix.sh).
