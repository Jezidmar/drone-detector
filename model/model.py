import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from functools import partial
import numpy as np
from model.model_core import (
    PatchEmbed_new,
    get_2d_sincos_pos_embed_flexible,
    FixedPositionalEncoder,
    AltBlock
)
from model.configuration_eat import EATConfig
import json
import tqdm

class EAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mode = "finetune"  # "pretrain" or "finetune"
        # === Embedding / Encoder ===
        self.local_encoder = PatchEmbed_new(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            stride=config.stride
        )

        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.drop_rate, inplace=True)
        trunc_normal_(self.extra_tokens, std=.02)

        self.fixed_positional_encoder = (
            FixedPositionalEncoder(self.build_sincos_pos_embed()) if config.fixed_positions else None
        )

        norm_layer = partial(nn.LayerNorm, eps=config.norm_eps, elementwise_affine=config.norm_affine)
        dpr = np.linspace(config.start_drop_path_rate, config.end_drop_path_rate, config.depth)
        self.blocks = nn.ModuleList([
            AltBlock(config.embed_dim, config.num_heads, config.mlp_ratio,
                     qkv_bias=config.qkv_bias, drop=config.drop_rate,
                     attn_drop=config.attn_drop_rate, mlp_drop=config.activation_dropout,
                     post_mlp_drop=config.post_mlp_drop, drop_path=dpr[i],
                     norm_layer=norm_layer, layer_norm_first=config.layer_norm_first,
                     ffn_targets=True)
            for i in range(config.depth)
        ])

        self.pre_norm = norm_layer(config.embed_dim)

        # === Head (for finetune) ===
        if self.mode == "finetune":
            self.fc_norm = nn.LayerNorm(config.embed_dim)
            self.head = nn.Linear(config.embed_dim, config.num_classes, bias=True)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def build_sincos_pos_embed(self):
        W = self.config.mel_bins // self.config.patch_size
        max_length = self.config.max_length
        embed_dim = self.config.embed_dim
        pos_embed = nn.Parameter(torch.zeros(1, max_length * W, embed_dim), requires_grad=False)
        emb = get_2d_sincos_pos_embed_flexible(embed_dim, (max_length, W), cls_token=False)
        pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))
        return pos_embed

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, x):
        B = x.shape[0]
        x = self.local_encoder(x)
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, None)[:, :x.size(1), :]
        x = torch.cat((self.extra_tokens.expand(B, -1, -1), x), dim=1)
        x = self.pre_norm(x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x, _ = blk(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        if self.mode == "finetune":
            x = x[:, 0]  # use cls token
            x = self.fc_norm(x)
            x = self.head(x)
        return x

    def extract_features(self, x):
        x = self.encode(x)
        return x