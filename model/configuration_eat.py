from transformers import PretrainedConfig


class EATConfig(PretrainedConfig):

    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        stride=16,
        in_chans=1,
        mel_bins=128,
        max_length=768,
        num_classes=527,
        model_variant="pretrain",  # or "finetune"
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        activation_dropout=0.0,
        post_mlp_drop=0.0,
        start_drop_path_rate=0.0,
        end_drop_path_rate=0.0,
        layer_norm_first=False,
        norm_eps=1e-6,
        norm_affine=True,
        fixed_positions=True,
        img_size=(1024, 128),  # (target_length, mel_bins)
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.mel_bins = mel_bins
        self.max_length = max_length
        self.num_classes = num_classes
        self.model_variant = model_variant

        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.activation_dropout = activation_dropout
        self.post_mlp_drop = post_mlp_drop
        self.start_drop_path_rate = start_drop_path_rate
        self.end_drop_path_rate = end_drop_path_rate

        self.layer_norm_first = layer_norm_first
        self.norm_eps = norm_eps
        self.norm_affine = norm_affine
        self.fixed_positions = fixed_positions

        self.img_size = img_size
