import pdb
import regex as re

from collections import defaultdict
import heapq


""" Data Structures
    word -> count           : static dictionary of words (from pretokenization) and how often they occur
    word -> tokens          : representation of each word as current list of tokens
    word -> pair_cnts       : pair counts of tokens representing word
    pair -> count           : number of times each pair occurs (over all words) -- dictionary and heap
    pair -> set[word]       : words in which the pairs occur
"""


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_file(input_path: str, verbose=False):
    # TODO: do this more efficiently
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
                if len(word) > 1:
                    for i in range(len(word) - 1):
                        pair = (word[i], word[i + 1])
                        word_to_pair_cnts[word][pair] += 1
                        occurrences[pair].add(word)
            word_counts[word] += 1
        
    return word_counts, word_to_tokens, word_to_pair_cnts, occurrences


def neg_pair(pair):
    return (-pair[0], -pair[1])


# input: list (split by special tokens) of iterators (given by pretokenization)
def initialize_bpe(docs: list, verbose=False):
    vocab = {i: chr(i).encode('utf-8') for i in range(256)}
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
        heapq.heappush(heap_counts, (-cnt, neg_pair(pair)))
    if verbose: print(f'Total Pair Counts: {heap_counts}')
    
    return vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences


def train_merge(best_pair, new_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences):
    words = pair_occurrences[best_pair].copy()
    updated_pairs = set()
    for word in words:
        for pair in word_to_pair_cnts[word]: 
            pair_counts[pair] -= (word_counts[word] * word_to_pair_cnts[word][pair])
            updated_pairs.add(pair)
            pair_occurrences[pair].remove(word)

        word_to_pair_cnts[word] = defaultdict(int)
        old_tokens = word_to_tokens[word]
        new_tokens = []
        i = 0
        while i < len(old_tokens):
            if i < len(old_tokens) - 1 and old_tokens[i] == best_pair[0] and old_tokens[i + 1] == best_pair[1]:
                new_tokens.append(new_vocab)
                i += 2
            else:
                new_tokens.append(old_tokens[i])
                i += 1
            if len(new_tokens) > 1: word_to_pair_cnts[word][(new_tokens[-2], new_tokens[-1])] += 1
        word_to_tokens[word] = new_tokens

        for pair in word_to_pair_cnts[word]:
            # if pair == (101, 116): pdb.set_trace()
            pair_counts[pair] += (word_counts[word] * word_to_pair_cnts[word][pair])
            updated_pairs.add(pair)
            pair_occurrences[pair].add(word)
    
    for pair in updated_pairs:
        heapq.heappush(heap_counts, (-pair_counts[pair], neg_pair(pair)))


def get_best_pair(pair_counts, heap_counts, verbose=False):
    best_count, neg_best_pair = heapq.heappop(heap_counts)
    best_pair = neg_pair(neg_best_pair)
    while pair_counts[best_pair] + best_count != 0:
        best_count, neg_best_pair = heapq.heappop(heap_counts)
        best_pair = neg_pair(neg_best_pair)
    
    best_count = - best_count
    return best_count, best_pair


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], verbose=False):
    file = read_file(input_path, verbose=verbose)
    chunks = split_special_tokens(file, special_tokens, verbose=verbose)
    docs = pretokenize(chunks, verbose=verbose)
    vocab,  word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences = initialize_bpe(docs, verbose=verbose)
    merges = []

    while len(vocab) < vocab_size:
        best_count, best_pair = get_best_pair(pair_counts, heap_counts, verbose=verbose)
        # if best_pair == (32, 67): pdb.set_trace()
        if best_count <= 1: 
            if verbose: print(f'No pairs with occurrence > 1')
            break
        if verbose:
            print(f'Merging {best_pair} {(vocab[best_pair[0]], vocab[best_pair[1]])} with value {best_count} to {len(vocab)}')

        new_vocab = len(vocab)
        vocab[new_vocab] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        train_merge(best_pair, new_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences)

    return vocab, merges


if __name__ == '__main__':
    test_special_tokens()
    print('---- TEST: Train BPE ----')
    vocab, merges = train_bpe('my_tests/sample.txt', 356, ['<|endoftext|>'], verbose=False)
    print(f'VOCABULARY: {vocab}')
    print(f'MERGES: {merges}')
