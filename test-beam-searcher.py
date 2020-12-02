import torch
from transformers import BertTokenizer, BertModel
from BeamSearcher import BeamSearcher
from format_attn_tuple import format_attn_tuple

torch.set_printoptions(profile="full")

#sentence_a = "Dylan is a songwriter"

sentence_a = input("Please input your sentence: \n")

print("\nReceive sentence: ", sentence_a, "\n")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer(sentence_a, return_tensors="pt", add_special_tokens=False)
outputs = model(**inputs, output_attentions=True, return_dict = False)

attention = outputs[-1]

searcher = BeamSearcher(tokenizer, inputs.input_ids, outputs[-1])

fact = searcher.search()

print("\nFound fact: ", " -> ".join(list(fact[0])), " Weight: ", fact[1], "\n")

print("\nCandidate Fact List: \n")

for f in searcher.candidates_history:
    print(f, "\n")



