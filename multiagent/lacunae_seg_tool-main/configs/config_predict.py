"""
Config file utility to ease the parameters change.
"""

class Config(object):

    test_params = {
        "patient_id": "all",     # Patient directory
        "nr_patches": 50,           # NR of patches per image row and column [30-50]
        "probab_th": .5,            # Probaility map threshold (DONOT CHANGE)
        "img_dim_row": 512,         # Row patch dimension to match model resolution
        "img_dim_col": 512          # Col patch dimension to match model resolution
    }

    model_params = {
        "model_name": "1hagjec2",   # Model file name (it corresponds to wandb run-id)
        "input_ch": 3,              # Image channel in input 
        "nr_classes": 1             # NR of classes to segment (lacunae == 1)
    }

    
