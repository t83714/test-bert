import torch
import transformers
from tabulate import tabulate
from typing import Dict, List, Tuple, Union
from operator import itemgetter, attrgetter


class SimpleSearcher:
    candidates_history: list = []

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, input_ids: torch.Tensor, attn_tuple: tuple, beam_size: int = 6):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.input_ids = input_ids[0].tolist()
        self.attn_matrix = self.create_weight_matrix(attn_tuple)
        table = tabulate(self.attn_matrix, tablefmt="fancy_grid")
        print("Attn Matrix: \n" + table + "\n")

    def create_weight_matrix(self, attn_tuple: tuple):
        last_layer: torch.Tensor = attn_tuple[-1][0]
        return torch.mean(last_layer, 0)

    def search(self):
        headIdx = torch.argmax(self.attn_matrix).item()
        headId = headIdx % self.attn_matrix.size()[0]
        headWeight = self.attn_matrix[headIdx//self.attn_matrix.size(
        )[0]][headIdx % self.attn_matrix.size()[0]].item()

        predictionWeightList = torch.t(self.attn_matrix)[headId][(headId+1):]
        predictionIdx = torch.argmax(predictionWeightList).item()
        predictionId = predictionIdx + headId + 1
        predictionWeight = predictionWeightList[predictionIdx].item()

        tailWeightList = self.attn_matrix.t()[headId][(predictionId+1):]
        tailIdx = torch.argmax(tailWeightList).item()
        tailId = tailIdx + predictionId + 1
        tailWeight = tailWeightList[tailIdx].item()

        return ((self.id_to_token(headId), headWeight), (self.id_to_token(predictionId), predictionWeight), (self.id_to_token(tailId), tailWeight))

    def id_to_token(self, id: int):
        return self.tokenizer.convert_ids_to_tokens(self.input_ids[id])

