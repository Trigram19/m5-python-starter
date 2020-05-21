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

from palsoftmax import *
from attention import *
from logsampler import *

class PositionwiseLSTMFF(torch.nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseLSTMFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), 
            nn.LSTMCell(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output
        
# Create Attention mask
def create_mask(qlen, mlen, dtype=torch.float32, same_length=False):
      """Creates attention mask when single-side context allowed only."""
      attn_mask = torch.ones([qlen, qlen], dtype=dtype)
      mask_u = torch.triu(attn_mask, 0, -1)
      mask_dia = torch.triu(attn_mask, 0, 0)
      attn_mask_pad = torch.zeros([qlen, mlen], dtype=dtype)
      ret = torch.concat([attn_mask_pad, mask_u - mask_dia], 1)
      if same_length:
        mask_l = torch.triu(attn_mask, -1, 0)
        ret = torch.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

      return ret
      
class PositionalEmbedding(Module):
  def __init__(self, dim, **kwargs):
    super(PositionalEmbedding, self).__init__(**kwargs)
    self.dim = dim

    """Constructs inversed frequency vector for positional embedding layer."""
    self.inv_freq = 1.0 / (10000.0**(torch.range(0, 19380, 10.0) / self.dim))

  def forward(self, pos_seq, batch_size):
    """Implements call() for the layer."""
    sinusoid_inp = torch.einsum('i,d->id', pos_seq, self.inv_freq)
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], -1)
    pos_emb = pos_emb[:, None, :]

    if batch_size is not None:
      pos_emb = tile(pos_emb, 2, self.dim)

    return pos_emb
    
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
    
## Took this from the TF GitHub repo and translated into Pytorch ##
class RelativeAttention(Module):
  """Core calculations for relative attention."""

  def __init__(self, dropout_att, scale):
    super(RelativeAttention, self).__init__()
    self.scale = scale
    self.dropout_att = dropout_att

  def build(self, unused_input_shapes):
    self.attention_probs_dropout = torch.nn.Dropout(
        p=self.dropout_att)

    super(RelativeAttention, self).build(unused_input_shapes)

  def call(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
           r_w_bias, r_r_bias, r_s_bias, attn_mask):
    # content based attention score
    ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

    # position based attention score
    bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
    bd = rel_shift(bd, klen=tf.shape(ac)[1])

    # segment-based attention score
    if seg_mat is None:
      ef = 0
    else:
      ef = torch.einsum('ibnd,snd->isbn', q_head + r_s_bias, seg_embed)
      tgt_shape = torch.shape(bd)
      ef = torch.where(
          torch.Tensor(np.broadcast_to(torch.expand_dims(seg_mat, 3), tgt_shape)),
          torch.Tensor(np.broadcast_to(ef[:, 1:, :, :], tgt_shape)),
          torch.Tensor(np.broadcast_to(ef[:, :1, :, :], tgt_shape)))

    # merges attention scores and performs masking
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
      attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = functional.softmax(attn_score, 1)
    attn_prob = self.attention_probs_dropout(attn_prob)

    # attention output
    attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
class MultiHeadAttn(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

class RelMultiHeadAttn(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError
def CBR(x, out_layer, kernel, stride, dilation):
    x = torch.nn.Conv1d(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = torch.nn.BatchNormalization()(x)
    x = torch.nn.functional.Activation("relu")(x)
    return x

def se_block(x_in, layer_n):
    x = torch.nn.GlobalAveragePooling1D()(x_in)
    x = torch.nn.Dense(layer_n//8, activation="relu")(x)
    x = torch.nn.Dense(layer_n, activation="sigmoid")(x)
    x_out=torch.nn.Multiply()([x_in, x])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = CBR(x_in, layer_n, kernel, 1, dilation)
    x = CBR(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = torch.nn.Add()([x_in, x])
    return x
class TimeDistributed(torch.nn.Module):
    def __init__(self, layer, time_steps, *args):        
        super(TimeDistributed, self).__init__()
        
        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):

        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([])
        for i in range(time_steps):
          output_t = self.layers[i](x[:, i, :, :, :])
          output_t  = y.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
        return output
        
class TransformerXLHybridEncoder(torch.nn.Module):
  def __init__(self,
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropout_att,
               attn_type,
               bi_data,
               is_training,
               initializer,
               mem_len=None,
               same_length=False,
               clamp_len=-1,
               untie_r=False,
               use_tpu=True,
               reuse_len=None,
               ff_activation='relu',
               use_cls_mask=False,
               **kwargs):

    super(TransformerXLHybridEncoder, self).__init__(**kwargs)

    self.n_token = n_token
    self.initializer = initializer
    self.attn_type = attn_type
    self.n_layer = n_layer
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.d_inner = d_inner
    self.ff_activation = ff_activation
    self.untie_r = untie_r
    self.use_tpu = use_tpu
    self.dropout = dropout
    self.dropout_att = dropout_att

    self.mem_len = mem_len
    self.reuse_len = reuse_len
    self.bi_data = bi_data
    self.clamp_len = clamp_len
    self.same_length = same_length
    self.use_cls_mask = use_cls_mask

  def build(self, unused_input_shapes):
    
    torch_float = torch.float32
    tf_float = torch_float
    self.inp = torch.nn.Input((28, -1))
    self.cbr = CBR(self.inp, 64, 7, 1, 1)
    self.embedding_lookup = EmbeddingLookup(
        n_token=self.n_token,
        d_embed=self.d_model,
        initializer=self.initializer,
        dtype=self.tf_float,
        name='embedding1')

    self.h_dropout = torch.nn.Dropout(p=self.dropout)
    self.g_dropout = torch.nn.Dropout(p=self.dropout)

    if self.untie_r:
      self.r_w_bias = (
          self.add_weight(
              'r_w_bias',
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_r_bias = (
          self.add_weight(
              'r_r_bias',
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_s_bias = (
          self.add_weight(
              'r_s_bias',
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
    else:
      self.r_w_bias = (
          self.add_weight(
              'r_w_bias',
              shape=[self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_r_bias = (
          self.add_weight(
              'r_r_bias',
              shape=[self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_s_bias = (
          self.add_weight(
              'r_s_bias', [self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))

    self.seg_embed = self.add_weight(
        'seg_embed', [self.n_layer, 2, self.n_head, self.d_head],
        dtype=self.torch_float,
        initializer=self.initializer)

    self.mask_emb = self.add_weight(
        'mask_emb/mask_emb', shape=[1, 1, self.d_model], dtype=self.torch_float)

    self.emb_dropout = torch.nn.Dropout(p=self.dropout)
    self.fwd_position_embedding = positionalencoding2d(self.d_model, self.n_head, self.d_head)
    self.fwd_td                 = TimeDistributed(self.fwd_position_embedding, time_steps=20)
    self.fwd_lstm               = torch.nn.LSTM(128)(self.fwd_td)
    self.hidden_vect_1 = (
        Variable(torch.zeros(1, 1, hidden_size)),
        Variable(torch.zeros(1, 1, hidden_size)))
    self.output1, self.hidden1 = self.fwd_lstm(Variable(torch.rand(1, 5, 10)), hidden_vect_1)
    self.bwd_position_embedding = PositionalEmbedding(self.d_model, self.n_head, self.d_head)
    self.bwd_td                 = TimeDistributed(self.bwd_position_embedding, time_steps=20)
    self.bwd_lstm               = torch.nn.LSTM(128)(self.bwd_td)
    self.hidden_position_embedding = PositionalEmbedding(self.d_model, self.n_head, self.d_head)
    self.hidden_td                 = TimeDistributed(self.hidden_position_embedding, time_steps=20)
    self.hidden_lstm               = torch.nn.LSTM(128)(self.hidden_td)
    self.hidden_vect_1 = (
        Variable(torch.zeros(1, 1, hidden_size)),
        Variable(torch.zeros(1, 1, hidden_size)))
    self.output2, self.hidden2 = self.bwd_lstm(Variable(torch.rand(1, 5, 10)), hidden_vect_1)

    self.rel_multihead_layers = []
    self.h_positionwise_ffn_layers = []
    self.layer_norm_layers = []
    for i in range(self.n_layer):
      self.rel_multihead_layers.append(
          RelMultiHeadAttn(
              d_model=self.d_model,
              dropout=self.dropout,
              n_head=self.n_head,
              d_head=self.d_head,
              name='layer_%d/rel_attn' % (i)))
      self.h_positionwise_ffn_layers.append(
          PositionwiseFF(
              d_model=self.d_model,
              d_inner=self.d_inner,
              dropout=self.dropout,
              kernel_initializer=self.initializer,
              activation_type=self.ff_activation,
              name='layer_%d/ff' % (i)))

    self.output_dropout = torch.nn.Dropout(p=self.dropout)
    
    def __call__(self,
               inp_k,
               seg_id=None,
               input_mask=None,
               mems=None,
               perm_mask=None,
               target_mapping=None,
               inp_q=None,
               **kwargs):
    # Uses dict to feed inputs into call() in order to keep mems as a python
    # list.
        inputs = {
        'inp_k': inp_k,
        'seg_id': seg_id,
        'input_mask': input_mask,
        'mems': mems,
        'perm_mask': perm_mask,
        'target_mapping': target_mapping,
        'inp_q': inp_q
    }
    return super(TransformerXLModel, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    inp_k = inputs['inp_k']
    seg_id = inputs['seg_id']
    input_mask = inputs['input_mask']
    mems = inputs['mems']
    perm_mask = inputs['perm_mask']
    target_mapping = inputs['target_mapping']
    inp_q = inputs['inp_q']

    new_mems = []

    bsz = torch.shape(inp_k)[1]

    qlen = inp_k.shape.as_list()[0]

    mlen = mems[0].shape.as_list()[0] if mems is not None else 0
    klen = mlen + qlen

    ##### Attention mask
    # causal attention mask
    if self.attn_type == 'uni':
      attn_mask = _create_mask(qlen, mlen, self.tf_float, self.same_length)
      # pylint: enable=protected-access
      attn_mask = attn_mask[:, :, None, None]
    elif self.attn_type == 'bi':
      attn_mask = None
    else:
      raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

    # data mask: input mask & perm mask
    if input_mask is not None and perm_mask is not None:
      data_mask = input_mask[None] + perm_mask

    elif input_mask is not None and perm_mask is None:
      data_mask = input_mask[None]
    elif input_mask is None and perm_mask is not None:
      data_mask = perm_mask
    else:
      data_mask = None

    if data_mask is not None:
      # all mems can be attended to
      mems_mask = torch.zeros([tf.shape(data_mask)[0], mlen, bsz],
                           dtype=self.tf_float)
      data_mask = torch.cat([mems_mask, data_mask], 1)
      if attn_mask is None:
        attn_mask = data_mask[:, :, :, None]
      else:
        attn_mask += data_mask[:, :, :, None]

    if attn_mask is not None:
      attn_mask = torch.cast(attn_mask > 0, dtype=self.tf_float)

    if attn_mask is not None:
      non_tgt_mask = -torch.eye(qlen, dtype=self.tf_float)
      non_tgt_mask = torch.cat(
          [tf.zeros([qlen, mlen], dtype=self.tf_float), non_tgt_mask], axis=-1)
      non_tgt_mask = torch.cast(
          (attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=self.tf_float)
    else:
      non_tgt_mask = None

    word_emb_k = self.embedding_lookup(inp_k)

    if inp_q is not None:
      if target_mapping is not None:
        word_emb_q = torch.tile(self.mask_emb,
                             [tf.shape(target_mapping)[0], bsz, 1])
      else:
        inp_q_ext = inp_q[:, :, None]
        word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k

    output_h = self.h_dropout(word_emb_k)
    output_g = None
    if inp_q is not None:
      output_g = self.g_dropout(word_emb_q)

    ##### Segment embedding
    if seg_id is not None:

      # Convert `seg_id` to one-hot `seg_mat`

      mem_pad = torch.zeros([mlen, bsz], dtype=tf.int32)

      cat_id = torch.concat([mem_pad, seg_id], 0)

      if self.use_cls_mask:
        # `1` indicates not in the same segment [qlen x klen x bsz]
        # seg_id: [qlen x bsz] & cat_id: [klen x bsz]
        cls_mat = torch.logical_or(
            torch.equal(seg_id, tf.constant([data_utils.SEG_ID_CLS]))[:, None],
            torch.equal(cat_id, tf.constant([data_utils.SEG_ID_CLS]))[None, :])
        seg_mat = torch.equal(seg_id[:, None], cat_id[None, :])
        seg_mat = torch.logical_or(cls_mat, seg_mat)
      else:
        seg_mat = tf.logical_not(tf.equal(seg_id[:, None], cat_id[None, :]))
    else:
      seg_mat = None

    dtype = self.tf_float
    freq_seq = tf.range(0, self.d_model, 2.0)
    if dtype is not None and dtype != tf.float32:
      freq_seq = tf.cast(freq_seq, dtype=self.dtype)

    if self.attn_type == 'bi':
      beg, end = klen, -qlen
    elif self.attn_type == 'uni':
      beg, end = klen, -1
    else:
      raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))

    if self.bi_data:
      fwd_pos_seq = torch.range(beg, end, -1.0)
      bwd_pos_seq = torch.range(-beg, -end, 1.0)

      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = torch.cast(fwd_pos_seq, dtype=dtype)
        bwd_pos_seq = torxh.cast(bwd_pos_seq, dtype=dtype)

      if self.clamp_len > 0:
        fwd_pos_seq = torch.clip_by_value(fwd_pos_seq, -self.clamp_len,
                                       self.clamp_len)
        bwd_pos_seq = torch.clip_by_value(bwd_pos_seq, -self.clamp_len,
                                       self.clamp_len)

      if bsz is not None:
        fwd_pos_emb = self.fwd_position_embedding(fwd_pos_seq, bsz // 2)
        bwd_pos_emb = self.bwd_position_embedding(bwd_pos_seq, bsz // 2)
      else:
        fwd_pos_emb = self.fwd_position_embedding(fwd_pos_seq, None)
        bwd_pos_emb = self.bwd_position_embedding(bwd_pos_seq, None)

      pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
      if self.clamp_len > 0:
        fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len,
                                       self.lamp_len)

      pos_emb = self.fwd_position_embedding(fwd_pos_seq, bsz)

    pos_emb = self.emb_dropout(pos_emb)

    if mems is None:
      mems = [None] * self.n_layer
    for i in range(self.n_layer):
      # cache new mems
      new_mems.append(
          _cache_mem(output_h, mems[i], self.mem_len, self.reuse_len))
      # pylint: enable=protected-access

      # segment bias
      if seg_id is None:
        r_s_bias_i = None
        seg_embed_i = None
      else:
        r_s_bias_i = self.r_s_bias if not self.untie_r else self.r_s_bias[i]
        seg_embed_i = self.seg_embed[i]

      ffn_layer = self.h_positionwise_ffn_layers[i]
      attention_layer = self.rel_multihead_layers[i]
      output_h, output_g = attention_layer(
          h=output_h,
          g=output_g,
          r=pos_emb,
          r_w_bias=self.r_w_bias if not self.untie_r else self.r_w_bias[i],
          r_r_bias=self.r_r_bias if not self.untie_r else self.r_r_bias[i],
          seg_mat=seg_mat,
          r_s_bias=r_s_bias_i,
          seg_embed=seg_embed_i,
          attn_mask_h=non_tgt_mask,
          attn_mask_g=attn_mask,
          mems=mems[i],
          target_mapping=target_mapping)
      output_h = ffn_layer(output_h)
      if output_g is not None:
        output_g = ffn_layer(output_g)

    if inp_q is not None:
      output = output_g
    else:
      output = output_h

    return output        
