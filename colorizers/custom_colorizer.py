import torch
from .eccv16 import eccv16
import os

def eccv16_custom(pretrained=False, weights_path=None):
    # pretrained=False skips downloading original weights
    model = eccv16(pretrained=pretrained)

    if weights_path is not None and os.path.exists(weights_path):
        print(f"Loading custom weights from {weights_path}")
        try:
            # Try loading with map_location to avoid device issues
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle both state_dict and full model save formats
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'module.' prefix if present (from DataParallel)
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
    elif weights_path is not None:
        print(f"Weights path {weights_path} does not exist. Using randomly initialized weights.")
        
    return model