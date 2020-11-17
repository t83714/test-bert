from transformers import BertTokenizer, BertModel
import torch

#torch.set_printoptions(profile="full")

sentence_a = "Dylan is a songwriter"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer(sentence_a, return_tensors="pt", add_special_tokens=False)
outputs = model(**inputs, output_attentions=True, return_dict = True)

print(outputs) 