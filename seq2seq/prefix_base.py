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

from prefix_tuning import PrefixTuningT5,PrefixTuning

logger = logging.getLogger(__name__)

class PrefixTransformer(PreTrainedModel):
    base_model_prefix = "prefixTransformer"

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

        if self.hparams.prefixModel_name_or_path is None:
            assert seq2seq_model is not None, "should initialize seq2seq_model"
        
        self.seq2seq_model = seq2seq_model

        config_prefix = AutoConfig.from_pretrained(self.hparams.model_name_or_path, cache_dir=cache_dir)
        self.model_type = self.config.model_type

        
        if self.hparams.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif self.hparams.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"




        print(self.model_type)
        config_prefix.prefixModel_name_or_path = self.hparams.prefixModel_name_or_path
        config_prefix._my_arg_tune_mode = self.hparams.tuning_mode
        config_prefix._my_arg_control = True
        config_prefix.train_weights = False
        config_prefix.optim_prefix = optim_prefix_bool
        config_prefix.preseqlen = self.hparams.preseqlen
        config_prefix.use_infix = (self.hparams.format_mode == 'infix')
        config_prefix.format_mode = self.hparams.format_mode
        config_prefix.prefix_dropout = self.hparams.prefix_dropout
        config_prefix.vocab_size = len(self.tokenizer)

        config_prefix.use_self_prefix = self.hparams.use_self_prefix
        config_prefix.use_cross_prefix = self.hparams.use_cross_prefix
        config_prefix.use_encoder_prefix = self.hparams.use_encoder_prefix
        config_prefix.init_train_epoch = self.hparams.init_train_epoch
        config_prefix.multi_languages = self.hparams.multi_languages.split('-') if self.hparams.multi_languages is not None else None
        if config_prefix.multi_languages is not None:
            config_prefix.multi_languages.sort()
        config_prefix.private_prefix = self.hparams.private_prefix 

        config_prefix.lowdata = self.hparams.lowdata#('lowdata' in self.hparams.output_dir)
        config_prefix.low_data_init = self.hparams.low_data_init
        if config_prefix.lowdata:
            config_prefix.lowdata_token = self.hparams.lowdata_token 
            config_prefix.lowdata_output_token = self.hparams.lowdata_output_token
        # some extra stuff.
        config_prefix.mid_dim = self.hparams.mid_dim
        
        # print(config_prefix)
        if self.hparams.prefixModel_name_or_path is not None and not self.hparams.load_whole_model:
            print('loading from {}'.format(hparams.prefixModel_name_or_path))
            if self.model_type == 'bart' or self.model_type == "mbart":
                self.model = PrefixTuning.from_pretrained(self.hparams.prefixModel_name_or_path,
                            from_tf=bool(".ckpt" in self.hparams.prefixModel_name_or_path),
                            cache_dir=cache_dir,
                            config=config_prefix,
                            model_gpt2=self.seq2seq_model)
            elif self.model_type == 'mt5' or self.model_type == 't5':
                self.model = PrefixTuningT5.from_pretrained(self.hparams.prefixModel_name_or_path,
                            from_tf=bool(".ckpt" in self.hparams.prefixModel_name_or_path),
                            cache_dir=cache_dir,
                            config=config_prefix,
                            model_gpt2=self.seq2seq_model)
            else:
                assert False, "do not support model type:{}".format(self.model_type)
        else:
            if self.model_type == "bart" or self.model_type == "mbart":
                self.model = PrefixTuning(config_prefix, self.seq2seq_model)
            elif self.model_type == "mt5" or self.model_type == 't5':
                self.model = PrefixTuningT5(config_prefix, self.seq2seq_model)
            else:
                assert False, "do not support model type:{}".format(self.model_type)
    
