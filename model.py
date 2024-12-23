"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right


@dataclass
class AttentionState:
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None

    def size(self):
        """Returns the size in bits"""
        return sum(t.numel() * t.element_size() for t in [self.k_cache, self.v_cache])


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # positional embeddings
        self.rotary_emb = RotaryEmbedding(dim=config.n_embd // config.n_head)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size



    def forward(self, x, state: AttentionState | None = None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Applying per-tensor affine quantization scheme for both k and v cache
        # S = (maxX - minX) / (int8_max - int8_min)
        # Z = round(-minX / S)
        # X_q = round(X / S) + Z
        # X = S * (X_q - Z)

        int8_max, int8_min  = 127, -128
        S_func = lambda t: torch.divide(torch.subtract(torch.max(t, dim=3, keepdim=True).values, torch.min(t, dim=3, keepdim=True).values), (int8_max - int8_min))
        Z_func = lambda t: torch.round(torch.divide(-torch.min(t, dim=3, keepdim=True).values, S_func(t)))

        # compute scale and zero point per token 
        S_k, Z_k = S_func(k), Z_func(k)
        S_v, Z_v = S_func(v), Z_func(v)

        if state is not None and state.k_cache is not None:
            # dequantize stored kv cache and append new kv
            k = torch.concat((torch.multiply(S_k, torch.subtract(state.k_cache, Z_k)).bfloat16(), k), dim=2)
            v = torch.concat((torch.multiply(S_v, torch.subtract(state.v_cache, Z_v)).bfloat16(), v), dim=2)
        
        # quantize and store new kv cache
        state = AttentionState(k_cache=torch.add(torch.round(torch.divide(k.to(torch.float16), S_k)), Z_k).detach(), v_cache=torch.add(torch.round(torch.divide(v.to(torch.float16), S_v)), Z_v).detach())

        # limit length of the entries in KV cache
        if k.size(2) > self.block_size:
          state.k_cache = state.k_cache[:,:,k.size(2)-self.block_size:k.size(2),:]
          state.v_cache=state.v_cache[:,:,k.size(2)-self.block_size:k.size(2),:]
  

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # when a state is provided we have more k and v entries than q entries,
        # in other words k.size(2) > q.size(2)
        # we construct a lower right mask matrix, which mathces all qs with the extra kvs:
        # For instance, with .size(2)=3, and k.size(2)=4, the mask is:
        # [[1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        attn_mask = causal_lower_right(q.size(2), k.size(2))

        # manual multi-head attention, useful for extracting attention scores
        #
        # attn_scores = torch.einsum("nhqd,nhkd->nqkh", [q, k])
        # attn_scores /= (q.size(-1) ** 0.5)
        # mask = attn_mask._materialize(torch.cuda.current_device())[None, :, :, None]
        # attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        # attn_scores = torch.softmax(attn_scores, dim=2)
        # y = torch.einsum("nqkh,nhkd->nqhd", [attn_scores, v]).reshape(B, T, C)

        # print(attn_scores.size())

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, state


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, state):
        y, state = self.attn(self.ln_1(x), state)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, state


@dataclass
class GPTConfig:
    vocab_size: int = 256  # Raw bytes
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    block_size: int = 256


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, block_states=None):
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        if block_states is None:
            block_states = [None] * len(self.transformer.h)
        new_block_states = []
        for block, block_state in zip(self.transformer.h, block_states):
            x, new_state = block(x, block_state)
            new_block_states.append(new_state)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_block_states

    def loss(self, logits, targets):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        block_states = [AttentionState() for _ in self.transformer.h]
        # put the first tokens to get initial state

        logits, block_states = self(idx, block_states=block_states)
        idxs = [idx]

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idxs.append(idx_next)
            logits, block_states = self(idx_next, block_states=block_states)

            # Assignment 1
            # ============
            # When the sequence length is longer than during training, the model produces
            # garbled output. This could be fixed by extrapolation of positional embeddings.
            # We'll take a different approach. Limit length of the entries in KV cache
            # to allow the model generate sequences longer than the training block size.

            # Assignment 2
            # ============
            # Implement as many tricks as you can think of to reduce the size of block_states.
            # Do not try to optimize the latency at all cost. Tip: pay attention to the offset
            # when applying rotary embeddings when the KV cache has been modified.
            #
            # Training-free ideas include:
            # - Quantizing the KV cache tensors to 8 bits. Which 8-bit format would you use?
            #   What it would take to go lower than 8 bits?
            # - Evicting (removing) unused tokens based on their cumulative attention scores
            #   (tech hint: the code for extracting attn scores is commented out)
            #
            # Retrain / do a few gradient steps from a saved checkpoint to change how
            # the model accesses its context:
            # - Do a few gradient steps with a much longer context. How many are needed to
            #   make it work?
            # - Use grouped query attention (have fewer KV heads that query heads)
            #   as in https://arxiv.org/abs/1911.02150 (tech hint: use the `enable_gqa`
            #   argument of scaled_dot_product_attention)
            # - Use long attention only in a few layers, limit others to small windows
            #   and share attention between neighboring layers
            #   https://research.character.ai/optimizing-inference/

            # How low, in terms of KV size in bits can you go (theoretically, you can
            # use masking and aligned data structures to make implementation easier)

        return torch.cat(idxs, dim=1)
