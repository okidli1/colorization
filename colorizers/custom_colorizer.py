import torch
import torch.nn as nn
import os
from .eccv16 import ECCVGenerator

def init_random_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -5.0, 5.0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def eccv16_custom(pretrained=False, weights_path=None):
    """
    Returns an ECCV16 model with:
      - pretrained weights if pretrained=True
      - custom weights if weights_path provided
      - random weights if neither is provided
    """
    model = ECCVGenerator()

    if weights_path is not None and os.path.exists(weights_path):
        print(f"Loading custom weights from {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            print("Custom weights loaded successfully!")

        except Exception as e:
            print(f"Error loading custom weights: {e}")
            print("Using randomly initialized weights instead.")
            model.apply(init_random_weights)
        return model

    if pretrained:
        import torch.utils.model_zoo as model_zoo
        url = 'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth'
        model.load_state_dict(model_zoo.load_url(url, map_location='cpu', check_hash=True))
        print("Loaded pretrained ECCV16 weights")
        return model

    print("Initializing ECCV16 model with RANDOM WEIGHTS")
    model.apply(init_random_weights)
    return model