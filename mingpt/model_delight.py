from mingpt.model import GPTConfig

import numpy as np

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class DelightConfig(GPTConfig):
    min_hidden_size = 128
    max_hidden_size = 2048

class GroupLinear(nn.Module):
    def __init__(self, in_features, out_features, n_groups=4, use_bias=True,
        use_shuffle=False, p_dropout=0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_groups = n_groups

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weight = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        self.norm = nn.LayerNorm(out_groups)
        self.dropout = nn.Dropout(p_dropout)
        self.act = nn.GELU()
        self.use_shuffle = use_shuffle
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def process_input_bmm(self, x):
        bsz = x.size(0)
        x = x.contiguous().view(bsz, self.n_groups, -1)
        x = x.transpose(0, 1)

        x = torch.bmm(x, self.weight)

        if self.bias is not None:
            x = torch.add(x, self.bias)

        if self.use_shuffle:
            x = x.permute(1, 2, 0)
            x = x.contiguous().view(bsz, self.n_groups, -1)
        else:
            x = x.transpose(0, 1)

        x = self.norm(x)
        x = self.act(x)

        return x

    def forward(self, x):
        if x.dim() == 2:
            x = self.process_input_bmm(x)
        elif x.dim() == 3:
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
        else:
            raise NotImplementedError

        x = self.dropout(x)

        return x

class DExTraUnit(nn.Module):

    def __init__(self, in_features, in_proj_features, out_features, n_layers=4, width_multiplier=2., max_groups=8, p_dropout=0.):
        super().__init__()

        self.n_layers = n_layers

        self.input_layer = nn.Sequential(
            nn.Linear(in_features, in_proj_features),
            nn.LayerNorm(in_proj_features),
            nn.GELU(),
            nn.Dropout(p_dropout),
        )
        dextra_config = self.dextra_config(
            in_proj_features,
            out_features,
            in_features * width_multiplier,
            n_layers,
            max_groups,
        )
        groups_next_layer = dextra_config['groups'][1:] + [1]
        self.dextra_layers = nn.ModuleList()
        
        for idx, (n_in, n_out, g_l, g_l1) in enumerate(zip(
            dextra_config['in'],
            dextra_config['out'],
            dextra_config['groups'],
            groups_next_layer
        )):
            if g_l == 1:
                wt_layer = nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.LayerNorm(n_out),
                    nn.GELU(),
                    nn.Dropout(p_dropout),
                )
            else:
                wt_layer = GroupLinear(
                    n_in,
                    n_out,
                    g_l,
                    use_shuffle=False,
                    p_dropout=p_dropout
                )
            self.dextra_layers.append(wt_layer)

        self.output_layer = nn.Sequential(
            nn.Linear(out_features + in_proj_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(p_dropout),
        )
        self.groups_per_layer = dextra_config['groups']

    def forward_dextra(self, x):
        '''
        T -- > time steps
        B --> Batch size
        N, M --> Input, output features
        :param x: Input is [TxBxN] or [BxTxN]
        :return: output is [TxBxM] or [BxTxM]
        '''
        B = x.size(0)
        T = x.size(1)

        out = x

        for i, layer_i in enumerate(self.dextra_layers):
            # Transform Layer
            out = layer_i(out)

            g_next_layer = self.groups_per_layer[i + 1] if i < self.n_layers - 1 else 1
            if g_next_layer == 1:
                # Linear layer is connected to everything so shuffle and split is useless for G=1
                out = torch.cat([x, out], dim=-1)
            else:
                # SPLIT and MIX LAYER
                # [B x T x M] --> [B x T x  G x M/G]
                x_g = x.contiguous().view(B, T, g_next_layer, -1)

                out = out.contiguous().view(B, T, g_next_layer, -1)

                # [B x T x G x M / G] || [B x T x G x N/G] --> [B x T x G x N+ M/G]
                out = torch.cat([x_g, out], dim=-1)

                # [B x T x G x N+ M/G] --> [B x T x N + M]
                out = out.contiguous().view(B, T, -1)

        out = self.output_layer(out)
        return out

    @staticmethod
    def dextra_config(in_features, out_features, max_features, n_layers, max_groups):

        mid_point = int(math.ceil(n_layers / 2.0))
        # decide number of groups per layer
        groups_per_layer = [min(2 ** (i + 1), max_groups) for i in range(mid_point)]

        # divide the space linearly between input_features and max_features
        output_sizes = np.linspace(in_features, max_features, mid_point, dtype=np.int).tolist()
        # invert lists to get the reduction groups and sizes
        inv_output_sizes = output_sizes[::-1]
        inv_group_list = groups_per_layer[::-1]
        if n_layers % 2 == 0:
            # even
            groups_per_layer = groups_per_layer + inv_group_list
            output_sizes = output_sizes + inv_output_sizes
        else:
            # for odd case,
            groups_per_layer = groups_per_layer + inv_group_list[1:]
            output_sizes = output_sizes + inv_output_sizes[1:]

        assert len(output_sizes) == len(groups_per_layer), '{} != {}'.format(len(output_sizes), len(groups_per_layer))
        output_sizes = output_sizes[:-1]

        # ensure that output and input sizes are divisible by group size
        input_sizes = [1] * len(groups_per_layer)
        input_sizes[0] = in_features
        for i in range(n_layers - 1):
            # output should be divisible by ith groups as well as i+1th group
            # Enforcing it to be divisble by 8 so that we can maximize tensor usage
            g_l = max(groups_per_layer[i + 1], groups_per_layer[i], 8)
            out_dim_l = int(math.ceil(output_sizes[i] / g_l)) * g_l
            inp_dim_l1 = out_dim_l + in_features

            if out_dim_l % 8 != 0:
                logger.warn(
                    'To maximize tensor usage, output dimension {} should be divisible by 8'.format(out_dim_l))

            if inp_dim_l1 % 8 != 0:
                logger.warn(
                    'To maximize tensor usage, input dimension {} should be divisible by 8'.format(inp_dim_l1))

            input_sizes[i + 1] = inp_dim_l1
            output_sizes[i] = out_dim_l

        # add dimensions corresponding to reduction step too
        output_sizes = output_sizes + [out_features]

        return {
            'in': input_sizes,
            'out': output_sizes,
            'groups': groups_per_layer
        }

    def forward(self, x):
        '''
        :param x: Input is [B x T x N]
        :return: Output is [B x T x M]
        '''

        # process input
        x = self.input_layer(x)
        n_dims = x.dim()

        if n_dims == 2:
            # [B x N] --> [B x 1 x N]
            x = x.unsqueeze(dim=1)  # add dummy T dimension
            # [B x 1 x N] --> [B x 1 x M]
            x = self.forward_dextra(x)
            # [B x 1 x M] --> [B x M]
            x = x.squeeze(dim=1)  # remove dummy T dimension
        elif n_dims == 3:
            x = self.forward_dextra(x)
        else:
            raise NotImplementedError
        return x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config, in_features, out_features):
        super().__init__()
        assert in_features % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(in_features, in_features)
        self.query = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(in_features, out_features)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch in_features
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class DelightBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, **dextra_kwargs):
        super().__init__()
        dextra_proj = 2
        proj_dim = config.n_embd // dextra_proj
        self.dextra_layer = DExTraUnit(
            config.n_embd,
            proj_dim,
            proj_dim,
            **dextra_kwargs,
        )
        self.attn = CausalSelfAttention(config, proj_dim, config.n_embd)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(config.n_embd // 4, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.dextra_layer(x))
        x = self.ln1(x)
        x = x + self.mlp(self.ln1(x))
        x = self.ln2(x)
        return x

class DelightModel(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        delight_dec_min_depth = 4
        delight_dec_max_depth = 8
        delight_dec_width_mult = 2

        dextra_depths = np.linspace(start=delight_dec_min_depth,
                                     stop=delight_dec_max_depth,
                                     num=config.n_layer,
                                     dtype=np.int)

        depth_ratio = (delight_dec_max_depth * 1.0) / delight_dec_min_depth
        width_multipliers = np.linspace(start=delight_dec_width_mult,
                                  stop=delight_dec_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
                                  num=config.n_layer,
                                  dtype=np.float
                                  )
        layers = []
        for ix in range(config.n_layer):
            layers.append(DelightBlock(config, n_layers=dextra_depths[ix], width_multiplier=width_multipliers[ix]))

        self.blocks = nn.Sequential(*layers)

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

