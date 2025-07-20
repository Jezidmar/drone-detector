import json
import torch
from model.model import EAT
from model.configuration_eat import EATConfig



def load_EAT_model(ckpt_path="EAT-large_epoch20_pt.pt", config_path="config.json"):
    """Loading model weights specifically for fine-tuning task on EAT architecture"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["num_classes"] = 1 # Binary classification task

    conf = EATConfig(**config)

    model = EAT(conf)

    # load pre-trained weights...
    state_dict = torch.load(
                ckpt_path
            )  # this should be loaded to CUDA:0 by default and but it does not matter since we transfer it to 'rank' device later on.


    model.load_state_dict(state_dict,strict=False)

    # freeze all weights from encoder.
    for key, val in model.named_parameters():
        if ("head" or "pre_norm") not in key:
            val.requires_grad = False

    return model