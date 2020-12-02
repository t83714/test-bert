import torch
import transformers
import nltk
from nltk.corpus import stopwords
from tabulate import tabulate
from typing import Dict, List, Tuple, Union
from operator import itemgetter, attrgetter
import re


class SimpleSearcher:
    """ a list of input id idx that has been picked or selected """
    selected_ids: list = []

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, input_ids: torch.Tensor, attn_tuple: tuple, beam_size: int = 6):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.input_ids = input_ids[0].tolist()
        self.attn_matrix = self.create_weight_matrix(attn_tuple)
        table = tabulate(self.attn_matrix, tablefmt="fancy_grid")
        #print("Attn Matrix: \n" + table + "\n")

    def create_weight_matrix(self, attn_tuple: tuple):
        last_layer: torch.Tensor = attn_tuple[-1][0]
        return torch.mean(last_layer, 0)

    def get_weight_matrix_row(self, row_num: int) -> list:
        return self.attn_matrix[row_num].tolist()

    def get_weight_matrix_col(self, col_num: int) -> list:
        return torch.t(self.attn_matrix)[col_num].tolist()

    def find_list_max_weight(self, l: list) -> Tuple[int, float]:
        return max(
            enumerate(l), key=itemgetter(1))

    def search(self, is_reverse: bool = False):
        return self.do_search(is_reverse)

    def do_search(self, is_reverse: bool = False):
        self.selected_ids = []

        head_str, head_id, head_weight = self.find_head_token()

        # search subwords
        head_str, head_subword_ids = self.look_for_sub_words(head_id)
        self.selected_ids.extend(head_subword_ids)

        prediction_str, prediction_id, prediction_weight = self.find_next_token(
            head_id, is_reverse)

        tail_str: str = None
        tail_id: int = None
        tail_weight: float = None

        if prediction_id is not None:
            # find tail
            tail_str, tail_id, tail_weight = self.find_next_token(
                prediction_id, is_reverse)

        return ((head_str, head_weight), (prediction_str, prediction_weight), (tail_str, tail_weight))

    def id_to_token(self, id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(self.input_ids[id])

    def look_for_sub_words(self, id: int):
        full_word_str = self.id_to_token(id)
        extra_ids = []
        # if origin token startWith '##'
        if full_word_str.startswith('##') == True and id != 0:
            for id in range(id-1, 0, -1):
                str = self.id_to_token(id)
                extra_ids.append(id)
                if str.startswith('##'):
                    full_word_str = str[2:] + full_word_str
                else:
                    full_word_str = str + full_word_str
                    break
        # search for sub words
        if id < len(self.input_ids) - 1:
            for id in range(id+1, len(self.input_ids)):
                str = self.id_to_token(id)
                if str.startswith('##') == False: break
                else:
                    extra_ids.append(id)
                    full_word_str += str[2:]
        return full_word_str, extra_ids

    def is_stop_token_id(self, id) -> bool:
        """Test id if it's a stop word token id or punctuation"""
        stop_words = {'the', 'a', 'an'}
        str = self.id_to_token(id)
        if str in stop_words:
            return True
        if re.search("[\da-zA-Z]+", str) is None:
            return True
        return False

    def find_max_weight_by_id_list(self, id_list: range, weight_list: list):
        """Find max weight id from id list base on weight list. Will skip stop words or any selected ids"""
        weight: float = 0
        select_id: int = None
        if len(id_list) == 0 or id_list is None:
            return None
        for id in id_list:
            if weight_list[id] >= weight and self.is_stop_token_id(id) == False and id not in self.selected_ids:
                weight = weight_list[id]
                select_id = id
        return select_id, weight

    def find_next_token(self, current_id: int, is_reverse: bool = False):
        total_token_num = len(self.input_ids)
        weightList = self.get_weight_matrix_col(current_id)
        id: int = None
        weight: float = None
        token_str: str = None
        if is_reverse == False and current_id < total_token_num - 1:
            # Search forward
            id, weight = self.find_max_weight_by_id_list(
                range(current_id + 1, total_token_num), weightList)
        if id is None and current_id > 0:
            # Search backward
            id, weight = self.find_max_weight_by_id_list(
                range(0, current_id), weightList)

        if id is not None:
            token_str, subword_ids = self.look_for_sub_words(id)
            self.selected_ids.append(id)
            self.selected_ids.extend(subword_ids)
        return token_str, id, weight

    def find_head_token(self):
        total_token_num = len(self.input_ids)
        weight:float = 0
        id: float = None
        token_str: str = None
        for x in range(total_token_num):
            w = self.attn_matrix[x][x].item()
            if w >= weight and self.is_stop_token_id(x) == False and id not in self.selected_ids:
                id = x
                weight = w
                token_str, subword_ids = self.look_for_sub_words(id)
                self.selected_ids.append(id)
                self.selected_ids.extend(subword_ids)
        return token_str, id, weight

