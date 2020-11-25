import torch
import transformers
import copy
from typing import Dict, List, Tuple, Union
from operator import itemgetter, attrgetter


class Candidate:
    head: Union[int, None] = None
    prediction: Union[int, None] = None
    tail: Union[int, None] = None
    weight: float = 0.0

    def __init__(self, weight: float = 0, head: Union[int, None] = None, tail: Union[int, None] = None, prediction: Union[int, None] = None) -> None:
        self.head = head
        self.prediction = prediction
        self.tail = tail
        self.weight = weight


class Fact:
    head: str = ""
    prediction: str = ""
    tail: str = ""
    weight: float = 0.0


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

# torch.mean(torch.stack([m1,m2]), 0)


class BeamSearcher:
    candidates_history:list = []

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, input_ids: torch.Tensor, attn_tuple: tuple, beam_size: int = 6):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.input_ids = input_ids[0].tolist()
        self.attn_matrix = self.create_weight_matrix(attn_tuple)

    def create_weight_matrix(self, attn_tuple: tuple):
        last_layer: torch.Tensor = attn_tuple[-1][0]
        return torch.mean(last_layer, 0)

    def get_weight(self, x: int, y: int, is_reverse: bool = False):
        if is_reverse == False:
            return self.attn_matrix[x][y].item()
        else:
            return self.attn_matrix[-x][-y].item()

    def get_min_candidates_weight(self):
        w = None
        idx = None
        for ci in range(len(self.candidates)):
            if w is None or w > self.candidates[ci].weight:
                idx = ci
                w = self.candidates[ci].weight
        return (idx, w)

    def search(self):
        self.candidates_history = []
        self.candidates: list[Candidate] = []
        fact = self.do_search()
        self.candidates_history += self.convert_candidate_list_to_tuple(self.candidates)

        self.candidates: list[Candidate] = []
        fact_rev = self.do_search(True)
        self.candidates_history += self.convert_candidate_list_to_tuple(self.candidates)

        if fact.weight >= fact_rev.weight:
            return self.convert_candidate_to_tuple(fact)
        else:
            return self.convert_candidate_to_tuple(fact_rev)

    def convert_candidate_list_to_tuple(self, c_list: List[Candidate], is_reverse: bool = False):
        result_list = []
        for item in c_list:
            result_list.append(
                self.convert_candidate_to_tuple(item, is_reverse))
        return result_list

    def convert_candidate_to_tuple(self, c: Candidate, is_reverse: bool = False):
        if is_reverse == False:
            return (
                (None if c.head is None else self.tokenizer.convert_ids_to_tokens(self.input_ids[c.head]),
                 None if c.prediction is None else self.tokenizer.convert_ids_to_tokens(
                 self.input_ids[c.prediction]),
                 None if c.tail is None else self.tokenizer.convert_ids_to_tokens(self.input_ids[c.tail])),
                c.weight
            )
        else:
            return (
                (None if c.head is None else self.tokenizer.convert_ids_to_tokens(self.input_ids[-c.head]),
                 None if c.prediction is None else self.tokenizer.convert_ids_to_tokens(
                 self.input_ids[-c.prediction]),
                 None if c.tail is None else self.tokenizer.convert_ids_to_tokens(self.input_ids[-c.tail])),
                c.weight
            )

    def do_search(self, is_reverse: bool = False):
        ids = self.input_ids[::-1] if is_reverse else self.input_ids
        for x in range(len(ids)):
            for y in range(len(ids)):
                self.search_with_ht(x, y, is_reverse)
        return self.search_predication(is_reverse)

    def search_with_ht(self, head_idx: int, tail_idx: int, is_reverse: bool):
        if head_idx == tail_idx:
            return
        weight = self.get_weight(head_idx, tail_idx, is_reverse)
        mw_idx, min_weight = self.get_min_candidates_weight()
        if min_weight is None or len(self.candidates) < self.beam_size:
            self.candidates.append(Candidate(weight, head_idx, tail_idx))
        elif min_weight <= weight:
            del self.candidates[mw_idx]
            self.candidates.append(Candidate(weight, head_idx, tail_idx))

    def search_predication(self, is_reverse: bool = False):
        ids = self.input_ids[::-1] if is_reverse else self.input_ids
        for p_idx in range(len(ids)):
            weight_list = [0]*len(self.candidates)
            for c_idx, c in enumerate(self.candidates):
                if p_idx == c.head or p_idx == c.tail:
                    continue
                weight = self.get_weight(
                    c.head, p_idx, is_reverse) + self.get_weight(p_idx, c.tail, is_reverse)
                weight_list[c_idx] = weight
            # pick a biggest weight
            biggest_idx, biggest_weight = max(
                enumerate(weight_list), key=itemgetter(1))
            self.candidates[biggest_idx].prediction = p_idx
            self.candidates[biggest_idx].weight = biggest_weight
        return max(self.candidates, key=attrgetter("weight"))
