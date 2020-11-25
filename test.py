import torch
from transformers import BertTokenizer, BertModel
from BeamSearcher import BeamSearcher
from format_attn_tuple import format_attn_tuple

torch.set_printoptions(profile="full")

m1 = torch.tensor([0.2, 0.4, 0.4])
m2 = torch.tensor([0.3, 0.5, 0.2])
mx = torch.mean(torch.stack([m1, m2]), 0)

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

searcher = BeamSearcher(tokenizer, inputs.input_ids, outputs[-1])

f = searcher.search()

print(f)

# format 12 of 1 x 12 x 4 x 4 tensor to 12 x 12 x 4 x 4 tensor
attn = format_attn_tuple(attention)

torch.tensor(attn.size())

print(attn) 


