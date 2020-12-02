import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from SimpleSearcher import SimpleSearcher
from format_attn_tuple import format_attn_tuple

torch.set_printoptions(profile="full")

gth_file = 'gth_part1'
sentences = pd.read_csv('{0}.csv'.format(gth_file))[
    'Sentences'].values.tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def process_sentence(sentence):
    print("Receive sentence: ", sentence, "\n")

    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    outputs = model(**inputs, output_attentions=True, return_dict=False)

    searcher = SimpleSearcher(tokenizer, inputs.input_ids, outputs[-1])

    fact = searcher.search()

    print("\nFound fact: ",
          " -> ".join(map(lambda item: item[0] + "("+str(item[1])+")", list(fact))), "\n")


for s in sentences:
    process_sentence(s)
