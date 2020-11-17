from transformers import BertTokenizer, BertModel
import torch

torch.set_printoptions(profile="full")


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

sentence_a = "Dylan is a songwriter"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer(sentence_a, return_tensors="pt", add_special_tokens=False)
outputs = model(**inputs, output_attentions=True, return_dict = False)

attention = outputs[-1]

# format 12 of 1 x 12 x 4 x 4 tensor to 12 x 12 x 4 x 4 tensor
attn = format_attention(attention)

print(attn) 