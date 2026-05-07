import pdb

import pickle
from pydoc import text
import regex as re

from collections import defaultdict
from collections.abc import Iterable, Iterator
import itertools


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BASE_VOCAB = 256


def split_special_tokens_keep(input: str, special_tokens: list, verbose=False):
    escaped_special_tokens = [re.escape(x) for x in special_tokens]
    delimiter = '(' + '|'.join(escaped_special_tokens) + ')'
    if verbose: print(f'Split Delimiter: {delimiter}')
    split_str = re.split(delimiter, input)
    return split_str


def get_counts(tokens):
    assert(len(tokens) > 1)
    pair_counts = defaultdict(int)
    for i in range(len(tokens) - 1):
        pair_counts[(tokens[i], tokens[i + 1])] += 1
    return pair_counts


def merge(pair, target, tokens):
    if len(tokens) <= 1:
        return tokens
    
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(target)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens
    

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        # pdb.set_trace()
        self.vocab = vocab
        if special_tokens is not None and len(special_tokens) > 0:
            special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
            added_tokens = 0
            for token in special_tokens:
                if token.encode('utf-8') not in vocab.values():
                    self.vocab[len(vocab) + added_tokens] = token.encode('utf-8')
                    added_tokens += 1
        # pdb.set_trace()
        self.special_tokens = special_tokens
        self.bytes_to_int = {byte_string: idx for idx, byte_string in self.vocab.items()}

        self.merges = {}
        for i, (p0, p1) in enumerate(merges):
            idx = BASE_VOCAB + i
            self.merges[(self.bytes_to_int[p0], self.bytes_to_int[p1])] = idx


    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = pickle.load(open(vocab_filepath, 'rb'))
        merges = pickle.load(open(merges_filepath, 'rb'))
        for token in special_tokens:
            idx = len(vocab)
            vocab[idx] = token.encode('utf-8')
        # TODO: what to do with this?


    def word_to_ints(self, word):
        result = []
        for c in word:
            encoding = list(c.encode('utf-8'))
            result.extend([self.bytes_to_int[bytes([i])] for i in encoding])
        return result

    
    def text_to_words(self, doc: str):
        list_of_words = []
        word_to_tokens = defaultdict(list)
        matches = re.finditer(PAT, doc)
        
        """
            list_of_words: list of words
            word_to_tokens: how each word is encoded
        """
        
        for i, match in enumerate(matches):
            word = match.group()
            list_of_words.append(word)
            if word not in word_to_tokens:
                word_to_tokens[word] = self.word_to_ints(word)

        return list_of_words, word_to_tokens
    

    def merge_one_word(self, tokens):
        while len(tokens) > 1:
            pair_counts = get_counts(tokens)
            pair = min(pair_counts, key=lambda p: self.merges.get(p, float('inf'))) # get first pair in merges that appears in tokens
            if pair not in self.merges:
                break # no more possible merges
            target = self.merges[pair]
            tokens = merge(pair, target, tokens)
        return tokens


    def encode(self, text: str) -> list[int]:
        if self.special_tokens is not None:
            docs = split_special_tokens_keep(text, self.special_tokens)
        else:
            docs = [text]

        result = []
        # pdb.set_trace()
        for doc in docs:
            # pdb.set_trace()
            if self.special_tokens is not None and doc in self.special_tokens:
                result.append(self.bytes_to_int[doc.encode('utf-8')])
            else:
                list_of_words, word_to_tokens = self.text_to_words(doc)
                for word in word_to_tokens.keys():
                    word_to_tokens[word] = self.merge_one_word(word_to_tokens[word])
                for word in list_of_words:
                    result.extend(word_to_tokens[word])
        # pdb.set_trace()
        return result


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        token_iterator = itertools.chain.from_iterable(map(self.encode, iterable))
        return token_iterator


    def decode(self, ids: list[int]) -> str:
        # pdb.set_trace()
        tokens = b''.join([self.vocab[id] for id in ids])
        text = tokens.decode('utf-8', errors='replace')
        return text


if __name__ == '__main__':
    pass