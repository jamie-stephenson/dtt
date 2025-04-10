import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from math import sqrt


class EmbeddingLayer(nn.Module):
    def __init__(
        self, vocab_size: int, n_ctx: int, d_model: int, device: str, dropout: float
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # (B,T) -> (B,T,C)
        self.positional_embedding = nn.Embedding(n_ctx, d_model)  # (T) -> (T,C)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        _, T = x.shape
        tok_embed = self.token_embedding(x)  # (B,T,C)
        pos_embed = self.positional_embedding(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = self.dropout(tok_embed + pos_embed)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        d_model: int,
        n_heads: int,
        device: str,
        dropout: float,
        mask_type: str,
        flash_attention: bool
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, device=device)
        # (B,T,C) -> (B,T,3C)

        self.out = nn.Linear(d_model, d_model)
        # (B,T,C) -> (B,T,C)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads

        if mask_type == "causal":
            self.register_buffer(
                "mask", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
            )
        else:
            self.register_buffer(
                "mask", torch.ones(n_ctx, n_ctx).view(1, 1, n_ctx, n_ctx)
            )

        self.flash_attention = flash_attention

    def forward(self, x):
        B, T, C = x.shape 

        # (B, T, C) -> (B, T, 3C) -> (B, T, C) x 3
        q, k, v = self.qkv(x).chunk(3, dim=-1) 

        # (B, T, C) -> (B, T, n_heads, C//n_heads) -> (B, n_heads, T, C//n_heads)
        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)

        if self.flash_attention:
            # (B, n_heads, T, C//n_heads) x 3 -> (B, n_heads, T, C//n_heads)
            v_attended_to = F.scaled_dot_product_attention(
                q, k, v, 
                is_causal=True, 
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )

        else:
            # (B, n_heads, T, T) -> (B, n_heads, T, T)
            attention_pattern = q @ k.transpose(-2, -1) / sqrt(C // self.n_heads)

            attention_pattern = attention_pattern.masked_fill(
                self.mask[:, :, :T, :T] == 0, -torch.inf
            )

            attention_pattern = self.attn_dropout(F.softmax(attention_pattern, dim=-1))

            # (B, n_heads, T, T) @ (B, n_heads, T, C//n_heads) -> (B, n_heads, T, C//n_heads)
            v_attended_to = attention_pattern @ v

        # (B, n_heads, T, C//n_heads) -> (B, T, n_heads, C//n_heads) -> (B, T, C)
        v_attended_to = v_attended_to.transpose(1, 2).contiguous().view(B, T, C)

        output = self.resid_dropout(self.out(v_attended_to))

        return output


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_mlp)
        self.layer_2 = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        d_mlp: int,
        d_model: int,
        n_heads: int,
        device: str,
        dropout: float,
        mask_type: str,
        flash_attention: bool
    ):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            n_ctx, d_model, n_heads, device, dropout, mask_type, flash_attention
        )

        self.layernorm_2 = nn.LayerNorm(d_model)

        self.feedforward = MLP(d_model, d_mlp, dropout)

    def forward(self, x):
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.feedforward(self.layernorm_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embed = EmbeddingLayer(
            cfg.vocab_size, cfg.n_ctx, cfg.d_model, cfg.device, cfg.dropout
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                cfg.n_ctx,
                cfg.d_mlp,
                cfg.d_model,
                cfg.n_heads,
                cfg.device,
                cfg.dropout,
                cfg.mask_type,
                cfg.flash_attention
            )
            for _ in range(cfg.n_blocks)
        ])

        # Note: no softmax on last layer, this model outputs logits ready for crossentropy loss.
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size)

        # init weights
        self.apply(self._init_weights)

        # weight tying
        self.unembed.weight = self.embed.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, EmbeddingLayer):
            # As per https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
            nn.init.normal_(module.token_embedding.weight, std=0.02)

        elif isinstance(module, MultiHeadAttention):
            nn.init.normal_(module.qkv.weight, std=1 / sqrt(module.qkv.in_features))
            nn.init.normal_(module.out.weight, std=1 / sqrt(module.out.in_features))

        elif isinstance(module, MLP):
            # As per https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/HookedTransformer.py#L1381
            nn.init.normal_(
                module.layer_1.weight, std=1 / sqrt(module.layer_1.in_features)
            )
            nn.init.normal_(
                module.layer_2.weight, std=1.57 / sqrt(module.layer_2.in_features)
            )


    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        x = self.unembed(x)

        return x


def get_model(cfg):
    model = Transformer(cfg)
    torch.set_float32_matmul_precision("high")

    model = torch.compile(
        DDP(model.to(cfg.device), cfg.device_id), mode="reduce-overhead"
    )

    return model
