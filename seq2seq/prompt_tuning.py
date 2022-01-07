# from transformers import Trainer
import torch
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer, PretrainedBartModel
from transformers import T5PreTrainedModel
from torch import  nn
import transformers
if transformers.__version__=="3.2.0":
    from transformers.modeling_bart import shift_tokens_right
else:
    from transformers.models.bart.modeling_bart import shift_tokens_right

import numpy as np
import random

# fix the random seed
def seed_everything(seed=11747):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class PromptTuningT5(T5PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, preseqlen=5,):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.num_decoder_layers
        self.match_n_head = config.num_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.use_encoder_prompt = config.use_encoder_prompt
        self.use_decoder_prompt = config.use_decoder_prompt


        if hasattr(config, 'preseqlen'):
            self.preseqlen = config.preseqlen
        else:
            self.preseqlen = preseqlen

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None

        if hasattr(config, 'lowdata_output_token'):
            self.lowdata_output_token = config.lowdata_output_token
        else:
            self.lowdata_output_token = None
  
        if config.multi_languages is not None and config.private_embedding:
            self.num_lang  = len(config.multi_languages)
        else:
            self.num_lang = 1

        print('preseqlen is {}, under the mode of optimizing prompt directly'.format(self.preseqlen))

        if self.use_encoder_prompt:
            self.wte_enc = nn.Embedding(self.preseqlen*self.num_lang, self.n_embd)
        if self.use_decoder_prompt:
            self.wte_dec = nn.Embedding(self.preseqlen*self.num_lang, self.n_embd)

        self.get_prompt = self.get_prompt

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))


    def get_prompt(self, bsz=None, lang_id = 0):
        self.input_tokens = torch.arange(self.preseqlen).long() + lang_id * self.preseqlen
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        
        token_embed_enc=None  
        token_embed_dec=None      

        if self.use_encoder_prompt:
            token_embed_enc = self.wte_enc(input_tokens)
        if self.use_decoder_prompt:
            token_embed_dec = self.wte_dec(input_tokens)        #[torch.Size([16, 200, 768])] bsz, num input_tokens, embd_size

        return token_embed_enc,token_embed_dec

    def get_encoder_output(self, gpt2, temp_input):
        return gpt2.encoder(temp_input,use_cache=True).past_key_values


    def forward(self,
        input_ids=None,
        gpt2_model=None,
        prompts=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        bsz = input_ids.shape[0]
        prompts = self.get_prompt(bsz=bsz, lang_id=kwargs['lang_id'] if 'lang_id' in kwargs else 0)

        if 'lang_id' in kwargs:
            kwargs.pop('lang_id') 

        prompts = [matrix.to(input_ids.device) if matrix is not None else None for matrix in prompts]

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        output = gpt2_model(input_ids=input_ids,
                            prompts=prompts, **kwargs)

        return output
