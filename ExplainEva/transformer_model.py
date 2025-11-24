import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class CustomMultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(CustomMultiheadAttention, self).__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        else:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter("in_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def merge_masks(self, attn_mask, key_padding_mask, query):
        mask_type = None
        merged_mask = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            batch_size, seq_len, _ = query.shape
            mask_type = 2
            if attn_mask.dim() == 3:
                merged_mask = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:
                merged_mask = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads,
                                                                                                    -1, -1)
                merged_mask = merged_mask + key_padding_mask_expanded

        return merged_mask, mask_type

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=False,
            is_causal=False,
            **kwargs
    ):
        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            batch_size, seq_len, embed_dim = query.size()
        else:
            seq_len, batch_size, embed_dim = query.size()
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        # Merge masks
        merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

        # Projections
        if self._qkv_same_embed_dim:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q = F.linear(query, self.q_proj_weight, self.q_proj_bias)
            k = F.linear(key, self.k_proj_weight, self.k_proj_bias)
            v = F.linear(value, self.v_proj_weight, self.v_proj_bias)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Add bias_k and bias_v if specified
        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(batch_size, self.num_heads, 1, 1)], dim=2)
            seq_len += 1
        if self.bias_v is not None:
            v = torch.cat([v, self.bias_v.repeat(batch_size, self.num_heads, 1, 1)], dim=2)

        # Add zero attention if specified
        if self.add_zero_attn:
            zero_v = torch.zeros(batch_size, self.num_heads, 1, self.head_dim, device=v.device)
            v = torch.cat([v, zero_v], dim=2)
            zero_k = torch.zeros(batch_size, self.num_heads, 1, self.head_dim, device=k.device)
            k = torch.cat([k, zero_k], dim=2)
            seq_len += 1

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device) * float('-inf'), diagonal=1)
            scores = scores + causal_mask

        if merged_mask is not None:
            scores = scores + merged_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        if not self.batch_first and is_batched:
            output = output.transpose(1, 0)

        # 始终返回 attn_weights 以便钩子捕获
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)
        return output, attn_weights


class Transformer(nn.Module):
    def __init__(self, cont_dim, discrete_dims, num_classes, hidden_dim=64, n_heads=8, n_layers=8):
        super(Transformer, self).__init__()
        self.cont_dim = cont_dim
        self.discrete_dims = discrete_dims
        self.hidden_dim = hidden_dim

        # Continuous feature embedding
        self.cont_embedding = nn.Linear(1, hidden_dim) if cont_dim > 0 else None

        # Discrete feature embeddings
        self.disc_embeddings = nn.ModuleList([
            nn.Embedding(n_levels, hidden_dim) for n_levels in discrete_dims
        ])

        # Positional encoding
        self.seq_len = len(discrete_dims) + cont_dim
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)

        # Transformer encoder with custom attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False
        )
        encoder_layer.self_attn = CustomMultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
            add_bias_kv=False,
            add_zero_attn=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                 norm=nn.LayerNorm(hidden_dim))

        # Output layer
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Register hooks for capturing attention weights
        self.attention_weights = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                self.attention_weights.append(output[1].detach().cpu().numpy())

        for layer in self.transformer.layers:
            layer.self_attn.register_forward_hook(hook)

    def forward(self, x_cont, x_disc):
        self.attention_weights = []  # 清空权重
        batch_size = x_cont.size(0) if self.cont_dim > 0 else x_disc.size(0)
        embeddings = []

        if self.cont_dim > 0:
            cont_embed = self.cont_embedding(x_cont.unsqueeze(-1))
            embeddings.append(cont_embed)
        for i, emb_layer in enumerate(self.disc_embeddings):
            disc_embed = emb_layer(x_disc[:, i])
            embeddings.append(disc_embed.unsqueeze(1))

        x = torch.cat(embeddings, dim=1)
        # print(x.shape)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

    def get_attention_weights(self):
        return self.attention_weights