import torch

# map each loss to which cfg‐fields it uses, and what argument name to give them
_PARAM_MAP = {
    'ce': {
        'use_class_weights': 'weight',
    },
    'bce': {
        'use_class_weights': 'weight',
    },
    'focal': {
        'focal_alpha':     'alpha',
        'focal_gamma':     'gamma',
        'focal_reduction': 'reduction',
    },
    'focalbce': {
        'focal_alpha':     'alpha',
        'focal_gamma':     'gamma',
        'focal_reduction': 'reduction',
        'bce_weight':      'bce_weight',
        'focal_weight':    'focal_weight',
    },
}

def loss_kwargs_manager(loss_cfg, class_weights=None):

    loss_name = loss_cfg.loss_function.lower()
    if loss_name not in _PARAM_MAP:
        raise KeyError(f"Unsupported loss: {loss_cfg.loss_function!r}")

    mapping = _PARAM_MAP[loss_name]
    kwargs  = {}

    for cfg_key, arg_name in mapping.items():

        if cfg_key == 'use_class_weights':
            
            if getattr(loss_cfg, 'use_class_weights', False):
                if class_weights is None:
                    raise ValueError("class_weights Series is required when use_class_weights=True")

                scaled = class_weights.pow(-loss_cfg.class_weights_temperature)
                weight_tensor = torch.tensor(
                    [ scaled[c] for c in sorted(scaled.index) ],
                    dtype=torch.float
                )
                kwargs[arg_name] = weight_tensor
        else:
            kwargs[arg_name] = getattr(loss_cfg, cfg_key)

    return kwargs
