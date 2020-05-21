import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc; import os
import torch
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
from tqdm.notebook import tqdm
from fastai.tabular import * 

from enc_and_utils import *
from palsoftmax import *
from attention import *
from logsampler import *
from decoders import *

class Transformer(torch.nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, attention_dropout_prob, output_dropout_prob, 
                 init_method, bi_data, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=384, ext_len=10, mem_len=384,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=2, clamp_len=-1,
                 sample_softmax=-1):
        super(Transformer, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.drop = nn.Dropout(dropout)

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type
        
        self.word_emb = PositionalEmbedding(d_model)

        self.layers = nn.ModuleList()
        self.layers.append(
            TransformerXLHybridEncoder(
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropatt,
               bi_data,
               attn_type,
               is_training=True,
               initializer=torch.optim.SGD,
            )
        )
        self.layers.append(  
            TransformerXLHybridEncoder(
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropatt,
               bi_data,
               attn_type,
               is_training=True,
               initializer=torch.optim.SGD,
            )
        )
        # the default attention
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        # learnable embeddings
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # absolute embeddings
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(
                    GPT2OptimizedDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, hidden_size=16)
                )
                self.layers.append (
                    GPT2OptimizedDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, hidden_size=16)
                )
        
        self.sample_softmax = sample_softmax
        self.out_layer = nn.Linear(d_model, n_token)
        self.sampler = LogUniformSampler(n_token, 1)
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
        
        
        
        # use adaptive softmax (including standard softmax)

        emb_layers = [i.weight for i in AdaptiveEmbedding(d_model, d_head, d_inner, n_head).emb_layers]
        emb_projs = AdaptiveEmbedding(d_model, d_head, d_inner, n_head).emb_projs

        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val,
                                                    tie_projs=tie_projs,
                                                    out_projs=emb_projs,
                                                    out_layers_weights=emb_layers)


        emb_projs = AdaptiveEmbedding(d_model, d_head, d_inner, n_head).emb_projs

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1
    cutoffs=[]
    
    def _create_params(self):
        # default attention
        cutoffs=[]
        if self.attn_type == 0:
            self.pos_emb = AdaptiveEmbedding(self.n_token, self.d_embed, self.d_model, cutoffs,
                                          div_val=self.div_val)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        # learnable
        elif self.attn_type == 1:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        # absolute standard
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        # absolute deeper SA
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.d_model))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        true_size = 7

        word_emb = PositionalEmbedding(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        # absolute
        if self.attn_type == 2:
            pos_seq = torch.LongTensor(torch.arange(klen - 1, -1, -1.0, dtype=torch.long))
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq, 64)

            core_out = self.drop(pos_emb[-qlen:])
            hids = []
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = core_out
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems

    def forward(self, data, target, crit, mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        self.criter = crit
        if mems is None:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=None)

        pred_hid = hidden[-tgt_len:]

        assert self.tie_weight
        criter = torch.nn.MSELoss()

        loss = criter(pred_hid.view(-1).reshape(512, 54292), target.view(-1))
        loss = loss.view(1, -1)
        return torch.Tensor(pred_hid)
