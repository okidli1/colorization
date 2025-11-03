import torch
from .eccv16 import eccv16

def eccv16_custom(pretrained=False, weights_path=None):
    # pretrained=False skips downloading original weights
    model = eccv16(pretrained=pretrained)

    if weights_path is not None:
        print(f"Loading custom weights from {weights_path}")
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        
    return model