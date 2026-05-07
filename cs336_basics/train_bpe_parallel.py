import pdb
import regex as re

from collections import defaultdict
import heapq

from cs336_basics.tokenizer import get_counts, merge
from cs336_basics.reverse import ReverseOrder


""" Data Structures
    word -> count           : static dictionary of words (from pretokenization) and how often they occur
    word -> tokens          : representation of each word as current list of tokens
    word -> pair_cnts       : pair counts of tokens representing word
    pair -> count           : number of times each pair occurs (over all words) -- dictionary and heap
    pair -> set[word]       : words in which the pairs occur
"""


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BASE_VOCAB = 256


def read_file(input_path: str, verbose=False):
    with open(input_path, 'r') as file:
        contents = file.read() # read in utf-8 encoding
    if verbose:
        print(contents)
    return contents


def split_special_tokens(input: str, special_tokens: list, verbose=False):
    escaped_special_tokens = [re.escape(x) for x in special_tokens]
    delimiter = '|'.join(escaped_special_tokens)
    if verbose: print(f'Split Delimiter: {delimiter}')
    split_str = re.split(delimiter, input)
    return split_str


def test_special_tokens():
    print('---- TEST: Split Special Tokens ----')
    delimiters = ['<|endoftext|>', '<|endofprompt|>']
    sample_str = 'abc<|endoftext|>def<|endofprompt|>xyz'
    test_output = split_special_tokens(sample_str, delimiters, verbose=True)
    output = ['abc', 'def', 'xyz']
    print(f'Output Match: {test_output == output}')


def pretokenize(inputs: list[str], verbose=False):
    iterators = [re.finditer(PAT, x) for x in inputs]
    return iterators


def init_counts(docs, verbose=False):
    word_counts = defaultdict(int)
    occurrences = defaultdict(set)
    word_to_tokens = {}
    word_to_pair_cnts = {}
    
    for doc in docs:
        for match in doc:
            word = match.group().encode('utf-8')
            if word not in word_counts:
                if verbose: print(f'Initializing Word: {word}')
                assert(word not in word_to_tokens)
                assert(word not in word_to_pair_cnts)
                tokens = list(word)
                word_to_tokens[word] = tokens
                word_to_pair_cnts[word] = defaultdict(int)
                if len(tokens) > 1:
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])
                        word_to_pair_cnts[word][pair] += 1
                        occurrences[pair].add(word)
            word_counts[word] += 1
        
    return word_counts, word_to_tokens, word_to_pair_cnts, occurrences


# given pair, return representation in bytes
def pair_to_comparator(pair, vocab):
    p0, p1 = pair
    # token -> byte string -> reverse
    c0 = ReverseOrder(vocab[p0])
    c1 = ReverseOrder(vocab[p1])
    return (c0, c1)


def comparator_to_pair(c_pair, rev_vocab):
    c0, c1 = c_pair
    # reverse -> byte string -> token
    p0 = rev_vocab[bytes(c0.get_val())]
    p1 = rev_vocab[bytes(c1.get_val())]
    return (p0, p1)


# input: list (split by special tokens) of iterators (given by pretokenization)
def initialize_bpe(docs: list, verbose=False):
    vocab = {i: bytes([i]) for i in range(BASE_VOCAB)}
    rev_vocab = {val: tok for tok, val in vocab.items()}

    word_counts, word_to_tokens, word_to_pair_cnts, pair_occurrences = init_counts(docs, verbose=verbose) 
    
    if verbose: 
        print(f'Word Counts: {word_counts}')
        print(f'Word To Pair Counts: {word_to_pair_cnts}')

    pair_counts = defaultdict(int)
    for pair, word_set in pair_occurrences.items():
        for word in word_set:
            pair_counts[pair] += (word_counts[word] * word_to_pair_cnts[word][pair])

    heap_counts = []
    for pair, cnt in pair_counts.items():
        heapq.heappush(heap_counts, (-cnt, pair_to_comparator(pair, vocab)))
    if verbose: print(f'Total Pair Counts: {heap_counts}')
    
    return vocab, rev_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences


def check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word):
    word_pair_cnts = get_counts(word_to_tokens[word])
    for pair in word_pair_cnts:
        total_count = pair_counts[pair]
        aggregate_count = sum([word_to_pair_cnts[w][pair] * word_counts[w] for w in pair_occurrences[pair]])
        if total_count != aggregate_count: 
            pdb.set_trace()
            print(f'Count Mismatch for Pair {pair}')
            return False
    return True


def train_merge(best_pair, new_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, pair_occurrences):
    words = pair_occurrences[best_pair].copy()   
    updated_pairs = set()
    for word in words:
        # debug_counts = get_counts(word_to_tokens[word])
        # pdb.set_trace()
        # assert(check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word)) 
        for pair, cnt in word_to_pair_cnts[word].items(): 
            # assert(cnt == debug_counts[pair])
            pair_counts[pair] -= (word_counts[word] * cnt)
            pair_occurrences[pair].remove(word)
            updated_pairs.add(pair)
        
        word_to_pair_cnts[word] = defaultdict(int)
        # assert(check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word))

        old_tokens = word_to_tokens[word]
        word_to_tokens[word] = merge(best_pair, new_vocab, old_tokens)
        if len(word_to_tokens[word]) > 1: 
            word_to_pair_cnts[word] = get_counts(word_to_tokens[word])
            for pair, cnt in word_to_pair_cnts[word].items():
                pair_counts[pair] += (word_counts[word] * cnt)
                pair_occurrences[pair].add(word)
                updated_pairs.add(pair)
            # assert(check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word))
    
    return updated_pairs


def get_best_pair(pair_counts, heap_counts, rev_vocab, verbose=False):
    best_count, neg_best_pair = heapq.heappop(heap_counts)
    best_pair = comparator_to_pair(neg_best_pair, rev_vocab)
    while pair_counts[best_pair] + best_count != 0:
        best_count, neg_best_pair = heapq.heappop(heap_counts)
        best_pair = comparator_to_pair(neg_best_pair, rev_vocab)
    
    best_count = - best_count
    return best_count, best_pair


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], verbose=False):
    file = read_file(input_path, verbose=verbose)
    chunks = split_special_tokens(file, special_tokens, verbose=verbose)
    docs = pretokenize(chunks, verbose=verbose)
    vocab, rev_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences = initialize_bpe(docs, verbose=verbose)
    merges = []

    while len(vocab) < vocab_size - len(special_tokens):
        best_count, best_pair = get_best_pair(pair_counts, heap_counts, rev_vocab, verbose=verbose)
        p0, p1 = best_pair
        if best_count <= 1: 
            if verbose: print(f'No pairs with occurrence > 1')
            break
        if verbose:
            print(f'Merging {best_pair} {(vocab[p0], vocab[p1])} with value {best_count} to {len(vocab)}')

        new_vocab = len(vocab)
        vocab[new_vocab] = vocab[p0] + vocab[p1]
        rev_vocab[vocab[p0] + vocab[p1]] = new_vocab
        merges.append((vocab[p0], vocab[p1]))
        updated_pairs = train_merge(best_pair, new_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, pair_occurrences)
        for pair in updated_pairs:
            heapq.heappush(heap_counts, (-pair_counts[pair], pair_to_comparator(pair, vocab)))

    for token in special_tokens:
        new_vocab = len(vocab)
        vocab[new_vocab] = token.encode('utf-8')
    
    return vocab, merges


if __name__ == '__main__':
    test_special_tokens()
    print('---- TEST: Train BPE ----')
    vocab, merges = train_bpe('my_tests/sample.txt', 356, ['<|endoftext|>'], verbose=False)
    print(f'VOCABULARY: {vocab}')
    print(f'MERGES: {merges}')
