# Multi-Sum
This is the Repo for the paper: Searching for Effective Multilingual Fine-Tuning Methods:A Case Study in Summarization.



## Table of Contents

- [Multi-Sum](#multi-sum)
  - [Table of Contents](#table-of-contents)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Benchmarks](#benchmarks)
  - [Training & Evaluation](#training--evaluation)

## Datasets
We use the XL-Sum corpus, which is a news dataset containing 1.35 million article-summary pairs in 45 languages. More details about the dataset can be found at [XL-Sum](https://github.com/csebuetnlp/xl-sum). The dataset can be downloaded from [XL-Sum](https://github.com/csebuetnlp/xl-sum) and [Huggingface](https://huggingface.co/datasets/csebuetnlp/xlsum). 

## Models
We are releasing two [multilingual model checkpoints](https://drive.google.com/drive/u/0/folders/1xp9gGw4gZXFLhCMJC0kXwFMjwJk8plL2) with best results mentioned in the paper. To use this model for evaluation/inference refer to [Training & Evaluation](#training--evaluation).
  
## Benchmarks
Multilingual model scores on test sets are given below. We are also releasing the [model-generated outputs](https://drive.google.com/drive/u/0/folders/1Do7cavlN-h6qVXzf2Wa3io7GoEzLC5y3) for future analysis.

Language |Multi-lingual Private-shared Adapter (R1/R2/RL)|Multi-lingual Private-shared Prefix (R1/R2/RL)
---------|-----------------------------|-----------------------------
Amharic | 20.6 / 7.61 / 18.5 | 20.33 / 7.48 / 18.34
Arabic | 35.16 / 14.96 / 29.27 | 35.36 / 15.11 / 29.49
Azerbaijani | 21.77 / 9.85 / 19.62 | 21.58 / 9.83 / 19.76
Bengali | 29.11 / 11.61 / 24.64 | 29.57 / 11.88 / 24.89
Burmese | 15.84 / 4.7 / 14.08 | 16.12 / 5.07 / 14.39
Chinese (Simplified) | 44.95 / 29.66 / 37.79 | 44.37 / 29.1 / 37.18
Chinese (Traditional) | 44.26 / 28.75 / 37.01 | 43.83 / 28.33 / 36.66
English | 38.29 / 15.71 / 30.41 | 38.29 / 15.63 / 30.39
French | 36.06 / 16.22 / 28.4 | 35.69 / 16.15 / 28.16
Gujarati | 22.45 / 7.97 / 20.21 | 22.38 / 8.14 / 20.31
Hausa | 39.75 / 17.64 / 31.85 | 39.63 / 17.8 / 31.96
Hindi | 39.18 / 17.37 / 32.49 | 39.02 / 17.3 / 32.35
Igbo | 30.17 / 9.2 / 22.98 | 30.11 / 9.5 / 23.1
Indonesian | 37.83 / 17.55 / 31.37 | 37.69 / 17.57 / 31.46
Japanese | 48.65 / 24.26 / 37.33 | 48.41 / 24.16 / 37.39
Kirundi | 32.65 / 14.98 / 26.22 | 32.65 / 14.91 / 26.26
Korean | 23.57 / 11.31 / 21.78 | 23.11 / 11.27 / 21.6
Kyrgyz | 18.4 / 7.72 / 16.0 | 18.3 / 7.8 / 16.11
Marathi | 23.13 / 10.32 / 20.8 | 22.82 / 10.02 / 20.54
Nepali | 26.5 / 10.05 / 24.06 | 26.59 / 10.2 / 24.21
Oromo | 19.68 / 6.45 / 16.91 | 19.49 / 6.71 / 16.88
Pashto | 38.85 / 15.85 / 32.05 | 38.92 / 16.05 / 32.16
Persian | 37.25 / 16.58 / 30.42 | 37.32 / 16.43 / 30.3
Pidgin | 38.56 / 15.6 / 30.14 | 38.89 / 15.88 / 30.47
Portuguese | 37.71 / 16.33 / 28.97 | 37.59 / 16.21 / 28.91
Punjabi | 31.14 / 12.3 / 25.33 | 30.77 / 12.27 / 25.29
Russian | 32.82 / 13.97 / 26.44 | 32.61 / 13.92 / 26.39
Scottish Gaelic | 28.85 / 10.55 / 22.65 | 30.08 / 11.2 / 23.83
Serbian (Cyrillic) | 24.5 / 8.42 / 20.73 | 24.4 / 8.35 / 20.62
Serbian (Latin) | 22.91 / 7.18 / 19.26 | 22.57 / 7.11 / 19.02
Sinhala | 27.7 / 13.8 / 23.61 | 27.43 / 13.41 / 23.7
Somali | 32.1 / 11.38 / 24.42 | 32.48 / 11.72 / 24.55
Spanish | 31.63 / 11.9 / 24.11 | 31.59 / 11.92 / 24.07
Swahili | 38.24 / 17.82 / 31.16 | 37.74 / 17.65 / 30.8
Tamil | 24.58 / 11.11 / 22.3 | 24.48 / 11.08 / 22.11
Telugu | 20.09 / 7.1 / 17.78 | 20.1 / 7.26 / 17.85
Thai | 37.84 / 17.34 / 28.81 | 37.99 / 17.65 / 29.07
Tigrinya | 25.85 / 8.51 / 21.6 | 25.48 / 8.55 / 21.78
Turkish | 33.63 / 16.17 / 29.95 | 33.58 / 16.1 / 29.81
Ukrainian | 24.73 / 10.7 / 21.58 | 24.75 / 10.62 / 21.57
Urdu | 40.04 / 18.71 / 33.15 | 40.12 / 18.71 / 33.21
Uzbek | 17.45 / 6.74 / 15.73 | 17.63 / 6.71 / 15.87
Vietnamese | 33.62 / 16.43 / 26.46 | 33.49 / 16.57 / 26.38
Welsh | 33.09 / 12.03 / 26.41 | 33.13 / 11.88 / 26.35
Yoruba | 31.95 / 11.92 / 25.24 | 31.71 / 11.61 / 24.91

## Training & Evaluation
  * See [training and evaluation module.](seq2seq/)



