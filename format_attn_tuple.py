import torch

# format 12 of 1 x 12 x 4 x 4 tensor to 12 x 12 x 4 x 4 tensor
def format_attn_tuple(atten_tuple: tuple):
    squeezed = []
    for layer_attention in atten_tuple:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                            "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)