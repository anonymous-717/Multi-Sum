import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

from torch import nn

from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from prompt_tuning import PromptTuningT5

logger = logging.getLogger(__name__)

class PromptTransformer(PreTrainedModel):
    base_model_prefix = "promptTransformer"

    def __init__(
        self,
        config=None,
        hparams=None,
        tokenizer=None,
        seq2seq_model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(config)

        self.hparams = hparams
        self.step_count = 0
        #self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        print('the cache dir is {}'.format(cache_dir))


        assert config is not None, "should initialize config"
        self.config: PretrainedConfig = config

        assert tokenizer is not None, "should initialize tokenizer"
        self.tokenizer: PreTrainedTokenizer = tokenizer

        if self.hparams.promptModel_name_or_path is None:
            assert seq2seq_model is not None, "should initialize seq2seq_model"
        
        self.seq2seq_model = seq2seq_model

        config_prompt = AutoConfig.from_pretrained(self.hparams.model_name_or_path, cache_dir=cache_dir)
        self.model_type = self.config.model_type


        print(self.model_type)
        config_prompt.promptModel_name_or_path = self.hparams.promptModel_name_or_path

        config_prompt.preseqlen = self.hparams.preseqlen
        config_prompt.use_encoder_prompt = self.hparams.use_encoder_prompt
        config_prompt.encoder_prompt_position = self.hparams.encoder_prompt_position
        config_prompt.use_decoder_prompt = self.hparams.use_decoder_prompt
        config_prompt.multi_languages = self.hparams.multi_languages.split('-') if self.hparams.multi_languages is not None else None
        if config_prompt.multi_languages is not None:
            config_prompt.multi_languages.sort()
        config_prompt.private_embedding = self.hparams.private_embedding 

        # print(config_prompt)
        if self.hparams.promptModel_name_or_path is not None and not self.hparams.load_whole_model:
            print('loading from {}'.format(hparams.promptModel_name_or_path))
            if self.model_type == 'mt5' or self.model_type == 't5':
                self.model = PromptTuningT5.from_pretrained(self.hparams.promptModel_name_or_path,
                            from_tf=bool(".ckpt" in self.hparams.promptModel_name_or_path),
                            cache_dir=cache_dir,
                            config=config_prompt,
                            model_gpt2=self.seq2seq_model)
            else:
                assert False, "do not support model type:{}".format(self.model_type)
        else:
            if self.model_type == "mt5" or self.model_type == 't5':
                self.model = PromptTuningT5(config_prompt, self.seq2seq_model)
            else:
                assert False, "do not support model type:{}".format(self.model_type)
    
